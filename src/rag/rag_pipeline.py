"""
rag_pipeline.py
---------------
RAG-augmented PR reviewer using a local Ollama backend.

At inference time:
  1. Strip the diff to added lines and retrieve relevant security documents
     from ChromaDB using nomic-embed-text embeddings + MMR.
  2. Inject the retrieved context into the prompt.
  3. Ask qwen2.5-coder to review the PR with citations to retrieved rules.
"""

import json
import requests

from tenacity import retry, wait_exponential, stop_after_attempt
from langchain_community.vectorstores import Chroma

from src.rag.vector_store import retrieve
from src.utils import get_logger

logger = get_logger(__name__)

_SYSTEM_PROMPT = """\
You are an expert security-focused code reviewer. You have been given relevant \
security rules and coding guidelines to assist your review.

When reviewing the pull request diff:
1. Summarize what the PR does in 2-3 sentences.
2. List concrete issues, especially security and correctness problems.  \
   For each issue: type, severity (low/medium/high), location, description, \
   and which rule or guideline it violates (cite the source by name).
3. Provide actionable suggestions for improvement.

Respond ONLY with a JSON object matching this schema:
{
  "summary": "<string>",
  "issues": [
    {
      "type": "<bug|security|style|performance|other>",
      "severity": "<low|medium|high>",
      "location": "<file:line or 'unknown'>",
      "description": "<string>",
      "rule_citation": "<name of the rule/guideline, or 'N/A'>"
    }
  ],
  "suggestions": ["<string>"],
  "retrieved_rules_used": ["<source name>"]
}
"""

_USER_TEMPLATE = """\
## Pull Request: {title}

### Relevant Security Rules & Guidelines
{context}

### Diff
```diff
{diff}
```
"""


def _diff_to_query(diff: str, max_chars: int = 2000) -> str:
    """
    Extract only the added lines from a diff for use as a retrieval query.
    Raw diffs contain noise (line numbers, deletions, markers) that dilutes
    the embedding signal. Added lines carry the intent of the change.
    Falls back to the raw diff if no added lines are found.
    """
    added = "\n".join(
        line[1:] for line in diff.splitlines()
        if line.startswith("+") and not line.startswith("+++")
    )
    query = added if added.strip() else diff
    return query[:max_chars]


class RAGReviewer:
    """PR reviewer that augments the prompt with retrieved security rules."""

    def __init__(self, config: dict, vector_store: Chroma):
        ollama_cfg = config["ollama"]
        rag_cfg = config["rag"]

        self.top_k = rag_cfg["top_k"]
        self.use_mmr = rag_cfg.get("use_mmr", True)
        self.mmr_lambda = rag_cfg.get("mmr_lambda", 0.6)
        self.store = vector_store

        self.model = ollama_cfg["model"]
        self._ollama_url = ollama_cfg["base_url"]
        self._ollama_stream = ollama_cfg.get("stream", False)
        self._max_diff_chars = ollama_cfg.get("max_diff_chars", 3000)
        self._temperature = ollama_cfg.get("temperature", 0.2)
        self._max_tokens = ollama_cfg.get("max_tokens", 1500)

        logger.info(f"RAGReviewer: model={self.model} top_k={self.top_k} mmr={self.use_mmr}")

    def _build_context(self, diff: str) -> tuple[str, list[str]]:
        """Retrieve docs and format them as a numbered list."""
        query = _diff_to_query(diff)
        docs = retrieve(
            self.store,
            query,
            top_k=self.top_k,
            use_mmr=self.use_mmr,
            mmr_lambda=self.mmr_lambda,
        )
        lines = []
        sources = []
        for i, doc in enumerate(docs, 1):
            source = doc.metadata.get("source", "Unknown")
            lines.append(f"{i}. [{source}]\n{doc.page_content.strip()}")
            sources.append(source)
        return "\n\n".join(lines), sources

    @retry(wait=wait_exponential(multiplier=2, min=4, max=60), stop=stop_after_attempt(4))
    def review(self, pr_record: dict) -> dict:
        diff = pr_record.get("diff", "")
        context, sources = self._build_context(diff)

        truncated_diff = diff[:self._max_diff_chars]
        user_msg = _USER_TEMPLATE.format(
            title=pr_record.get("title", "Untitled"),
            context=context,
            diff=truncated_diff,
        )
        prompt = f"{_SYSTEM_PROMPT}\n\n{user_msg}"

        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": self._ollama_stream,
            "format": "json",
            "options": {
                "temperature": self._temperature,
                "num_predict": self._max_tokens,
            },
        }

        resp = requests.post(self._ollama_url, json=payload, timeout=180)
        resp.raise_for_status()
        raw = resp.json().get("response", "")

        try:
            review_json = json.loads(raw)
        except json.JSONDecodeError:
            logger.warning(f"PR {pr_record.get('pr_id')}: Ollama returned invalid JSON — storing raw")
            review_json = {"raw": raw}

        return {
            "pr_id": pr_record.get("pr_id"),
            "repo": pr_record.get("repo"),
            "model": self.model,
            "approach": "rag",
            "review": review_json,
            "retrieved_sources": sources,
        }
