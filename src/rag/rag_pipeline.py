"""
rag_pipeline.py
---------------
RAG-augmented PR reviewer.

At inference time:
  1. Retrieve relevant security / coding rules from ChromaDB using the diff as
     the query.
  2. Inject the retrieved context into the prompt.
  3. Ask the LLM to review the PR with citations to the retrieved rules.

Supports two backends, selected automatically from config:
  - Ollama  (config["ollama"] present and api_key omitted)
  - OpenAI-compatible endpoint  (Gemini, OpenAI, etc.)
"""

import json
import requests

from openai import OpenAI
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


class RAGReviewer:
    """PR reviewer that augments the prompt with retrieved security rules."""

    def __init__(self, config: dict, vector_store: Chroma, api_key: str = ""):
        llm_cfg = config["llm"]
        ollama_cfg = config.get("ollama")

        # Use Ollama when the config section exists and no API key is supplied
        self._use_ollama = bool(ollama_cfg) and not api_key
        self.top_k = config["rag"]["top_k"]
        self.store = vector_store
        self.temperature = llm_cfg.get("temperature", 0.2)
        self.max_tokens = llm_cfg.get("max_tokens", 1500)

        if self._use_ollama:
            self.model = ollama_cfg["model"]
            self._ollama_url = ollama_cfg["base_url"]
            self._ollama_stream = ollama_cfg.get("stream", False)
            self._max_diff_chars = ollama_cfg.get("max_diff_chars", 3000)
            logger.info(f"RAGReviewer using Ollama: {self._ollama_url} model={self.model}")
        else:
            self.model = llm_cfg["model"]
            base_url = llm_cfg.get("base_url")
            self.client = OpenAI(api_key=api_key, base_url=base_url)
            logger.info(f"RAGReviewer using OpenAI-compat endpoint: model={self.model}")

    def _build_context(self, diff: str) -> tuple[str, list[str]]:
        """Retrieve docs and format them as a numbered list."""
        docs = retrieve(self.store, diff, top_k=self.top_k)
        lines = []
        sources = []
        for i, doc in enumerate(docs, 1):
            source = doc.metadata.get("source", "Unknown")
            lines.append(f"{i}. [{source}]\n{doc.page_content.strip()}")
            sources.append(source)
        return "\n\n".join(lines), sources

    # ── Ollama backend ────────────────────────────────────────────────────────

    def _ollama_review(self, pr_record: dict, context: str, sources: list[str]) -> dict:
        diff = pr_record.get("diff", "")[:self._max_diff_chars]
        user_msg = _USER_TEMPLATE.format(
            title=pr_record.get("title", "Untitled"),
            context=context,
            diff=diff,
        )
        prompt = f"{_SYSTEM_PROMPT}\n\n{user_msg}"

        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": self._ollama_stream,
            "format": "json",
        }

        resp = requests.post(self._ollama_url, json=payload, timeout=180)
        resp.raise_for_status()
        raw = resp.json().get("response", "")

        try:
            review_json = json.loads(raw)
        except json.JSONDecodeError:
            logger.warning(f"PR {pr_record.get('pr_id')}: Ollama returned invalid JSON -- storing raw")
            review_json = {"raw": raw}

        return {
            "pr_id": pr_record.get("pr_id"),
            "repo": pr_record.get("repo"),
            "model": self.model,
            "approach": "rag",
            "review": review_json,
            "retrieved_sources": sources,
        }

    # ── OpenAI-compatible backend ─────────────────────────────────────────────

    def _openai_review(self, pr_record: dict, context: str, sources: list[str]) -> dict:
        user_msg = _USER_TEMPLATE.format(
            title=pr_record.get("title", "Untitled"),
            context=context,
            diff=pr_record.get("diff", ""),
        )

        response = self.client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            response_format={"type": "json_object"},
        )

        raw = response.choices[0].message.content
        try:
            review_json = json.loads(raw)
        except json.JSONDecodeError:
            logger.warning(f"PR {pr_record.get('pr_id')}: LLM returned invalid JSON")
            review_json = {"raw": raw}

        return {
            "pr_id": pr_record.get("pr_id"),
            "repo": pr_record.get("repo"),
            "model": self.model,
            "approach": "rag",
            "review": review_json,
            "retrieved_sources": sources,
        }

    # ── Public interface ──────────────────────────────────────────────────────

    @retry(wait=wait_exponential(multiplier=2, min=4, max=60), stop=stop_after_attempt(4))
    def review(self, pr_record: dict) -> dict:
        diff = pr_record.get("diff", "")
        context, sources = self._build_context(diff)

        if self._use_ollama:
            return self._ollama_review(pr_record, context, sources)
        return self._openai_review(pr_record, context, sources)

    def review_batch(self, records: list[dict]) -> list[dict]:
        results = []
        for record in records:
            logger.info(f"RAG review: PR #{record.get('pr_id')}")
            results.append(self.review(record))
        return results
