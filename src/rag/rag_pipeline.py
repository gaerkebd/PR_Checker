"""
rag_pipeline.py
---------------
RAG-augmented PR reviewer.

At inference time:
  1. Retrieve relevant security / coding rules from ChromaDB using the diff as
     the query.
  2. Inject the retrieved context into the prompt.
  3. Ask the LLM to review the PR with citations to the retrieved rules.
"""

import json

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

    def __init__(self, config: dict, vector_store: Chroma, api_key: str):
        llm_cfg = config["llm"]
        self.model = llm_cfg["model"]
        self.temperature = llm_cfg.get("temperature", 0.2)
        self.max_tokens = llm_cfg.get("max_tokens", 1500)
        self.top_k = config["rag"]["top_k"]
        self.store = vector_store
        self.client = OpenAI(api_key=api_key)

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

    @retry(wait=wait_exponential(multiplier=2, min=4, max=60), stop=stop_after_attempt(4))
    def review(self, pr_record: dict) -> dict:
        """
        Generate a RAG-augmented code review for a single preprocessed PR record.

        Returns
        -------
        dict with keys: pr_id, repo, review, model, approach, retrieved_sources
        """
        diff = pr_record.get("diff", "")
        context, sources = self._build_context(diff)

        user_msg = _USER_TEMPLATE.format(
            title=pr_record.get("title", "Untitled"),
            context=context,
            diff=diff,
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

    def review_batch(self, records: list[dict]) -> list[dict]:
        results = []
        for record in records:
            logger.info(f"RAG review: PR #{record.get('pr_id')}")
            results.append(self.review(record))
        return results
