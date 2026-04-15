"""
evaluator.py
------------
Evaluates generated PR reviews against human reviewer comments.

Metrics:
  1. Semantic Similarity   — cosine similarity between review embeddings
  2. ROUGE-L               — recall-oriented n-gram overlap
  3. Issue Detection Score — heuristic: do review issues mention the same
                             keywords as the human comments?
  4. Hallucination Check   — do cited rules appear in retrieved_sources?
"""

import json
import os
import re
from typing import Optional

import numpy as np
from rouge_score import rouge_scorer
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from src.utils import get_logger

logger = get_logger(__name__)

_ROUGE = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)


class Evaluator:
    def __init__(self, embedding_model: str = "all-MiniLM-L6-v2"):
        logger.info(f"Loading embedding model: {embedding_model}")
        self.embedder = SentenceTransformer(embedding_model)

    # ── Semantic similarity ──────────────────────────────────────────────────

    def semantic_similarity(self, generated: str, reference: str) -> float:
        """Cosine similarity between two text strings (0–1)."""
        if not generated or not reference:
            return 0.0
        vecs = self.embedder.encode([generated, reference], normalize_embeddings=True)
        return float(cosine_similarity([vecs[0]], [vecs[1]])[0][0])

    # ── ROUGE-L ──────────────────────────────────────────────────────────────

    def rouge_l(self, generated: str, reference: str) -> float:
        """ROUGE-L F1 score."""
        if not generated or not reference:
            return 0.0
        score = _ROUGE.score(reference, generated)
        return score["rougeL"].fmeasure

    # ── Issue detection (keyword heuristic) ──────────────────────────────────

    @staticmethod
    def _keywords(text: str) -> set[str]:
        """Lower-cased alphanumeric tokens longer than 3 chars."""
        return {t for t in re.findall(r"[a-zA-Z0-9_]+", text.lower()) if len(t) > 3}

    def issue_detection_score(self, generated_review: dict, human_comments: list[str]) -> float:
        """
        Fraction of keywords that appear in human comments that also appear in
        the generated review issues/suggestions.
        Returns a float in [0, 1].
        """
        human_text = " ".join(human_comments)
        human_kw = self._keywords(human_text)
        if not human_kw:
            return 0.0

        # Flatten generated issues + suggestions into one string
        issues = generated_review.get("issues", [])
        suggestions = generated_review.get("suggestions", [])
        gen_text = " ".join(
            [i.get("description", "") for i in issues] + suggestions
        )
        gen_kw = self._keywords(gen_text)

        overlap = human_kw & gen_kw
        return len(overlap) / len(human_kw)

    # ── Hallucination check ───────────────────────────────────────────────────

    @staticmethod
    def hallucination_score(generated_review: dict, retrieved_sources: list[str]) -> float:
        """
        For RAG results: fraction of rule_citations in generated issues that
        actually appear in the retrieved sources.  1.0 = no hallucinated citations.
        Returns None for baseline (no retrieved sources).
        """
        if not retrieved_sources:
            return None  # not applicable for baseline

        citations = [
            i.get("rule_citation", "N/A")
            for i in generated_review.get("issues", [])
            if i.get("rule_citation") and i["rule_citation"] != "N/A"
        ]
        if not citations:
            return 1.0  # no citations made → nothing hallucinated

        sources_lower = [s.lower() for s in retrieved_sources]
        grounded = sum(
            1 for c in citations
            if any(c.lower() in s for s in sources_lower)
        )
        return grounded / len(citations)

    # ── Full evaluation ───────────────────────────────────────────────────────

    def evaluate_single(self, result: dict, processed_record: dict) -> dict:
        """
        Evaluate one model output against its corresponding processed PR record.

        Parameters
        ----------
        result           : output from BaselineReviewer.review() or RAGReviewer.review()
        processed_record : record from the preprocessed dataset

        Returns
        -------
        dict of metric scores
        """
        review = result.get("review", {})
        human_comments = processed_record.get("human_review", [])
        retrieved_sources = result.get("retrieved_sources", [])

        # Flatten generated review to text for similarity metrics
        gen_summary = review.get("summary", "")
        gen_issues_text = " ".join(
            i.get("description", "") for i in review.get("issues", [])
        )
        gen_text = f"{gen_summary} {gen_issues_text}".strip()
        human_text = " ".join(human_comments)

        sem_sim = self.semantic_similarity(gen_text, human_text)
        rouge = self.rouge_l(gen_text, human_text)
        issue_det = self.issue_detection_score(review, human_comments)
        halluc = self.hallucination_score(review, retrieved_sources)

        return {
            "pr_id": result.get("pr_id"),
            "approach": result.get("approach"),
            "model": result.get("model"),
            "semantic_similarity": round(sem_sim, 4),
            "rouge_l": round(rouge, 4),
            "issue_detection_score": round(issue_det, 4),
            "hallucination_score": round(halluc, 4) if halluc is not None else None,
            "n_issues_generated": len(review.get("issues", [])),
            "n_human_comments": len(human_comments),
        }

    def evaluate_batch(
        self,
        results: list[dict],
        processed_records: list[dict],
    ) -> tuple[list[dict], dict]:
        """
        Evaluate all model results.

        Returns
        -------
        (per_pr_scores, aggregate_summary)
        """
        # Build lookup by pr_id
        record_map = {r["pr_id"]: r for r in processed_records}

        scores = []
        for res in results:
            pr_id = res.get("pr_id")
            rec = record_map.get(pr_id)
            if rec is None:
                logger.warning(f"No processed record found for PR #{pr_id} — skipping.")
                continue
            scores.append(self.evaluate_single(res, rec))

        if not scores:
            return scores, {}

        # Aggregate per approach
        aggregate: dict[str, dict] = {}
        for score in scores:
            approach = score["approach"]
            if approach not in aggregate:
                aggregate[approach] = {
                    "semantic_similarity": [],
                    "rouge_l": [],
                    "issue_detection_score": [],
                    "hallucination_score": [],
                }
            for key in aggregate[approach]:
                val = score[key]
                if val is not None:
                    aggregate[approach][key].append(val)

        summary = {
            approach: {k: round(float(np.mean(v)), 4) if v else None for k, v in metrics.items()}
            for approach, metrics in aggregate.items()
        }

        logger.info("Evaluation complete. Aggregate results:")
        for approach, metrics in summary.items():
            logger.info(f"  [{approach}] {metrics}")

        return scores, summary


def save_results(per_pr: list[dict], summary: dict, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    output = {"summary": summary, "per_pr": per_pr}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    logger.info(f"Evaluation results saved → {path}")
