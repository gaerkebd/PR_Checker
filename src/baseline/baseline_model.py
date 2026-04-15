"""
baseline_model.py
-----------------
Zero-retrieval LLM reviewer.  Takes a PR diff and returns structured JSON
containing a summary, detected issues, and improvement suggestions.
"""

import json
from typing import Optional

from openai import OpenAI
from tenacity import retry, wait_exponential, stop_after_attempt

from src.utils import get_logger, load_config

logger = get_logger(__name__)

_SYSTEM_PROMPT = """\
You are an expert code reviewer. When given a pull request diff you must:
1. Summarize what the PR does in 2-3 sentences.
2. List concrete issues found (bugs, security risks, style violations). \
   For each issue include: type, severity (low/medium/high), location (file and line if visible), and a brief description.
3. Provide actionable suggestions for improvement.

Respond ONLY with a JSON object matching this schema exactly:
{
  "summary": "<string>",
  "issues": [
    {
      "type": "<bug|security|style|performance|other>",
      "severity": "<low|medium|high>",
      "location": "<file:line or 'unknown'>",
      "description": "<string>"
    }
  ],
  "suggestions": ["<string>"]
}
"""

_USER_TEMPLATE = """\
## Pull Request: {title}

### Diff
```diff
{diff}
```
"""


class BaselineReviewer:
    """Wraps an OpenAI-compatible chat endpoint for zero-retrieval PR review."""

    def __init__(self, config: dict, api_key: str):
        llm_cfg = config["llm"]
        self.model = llm_cfg["model"]
        self.temperature = llm_cfg.get("temperature", 0.2)
        self.max_tokens = llm_cfg.get("max_tokens", 1500)
        self.client = OpenAI(api_key=api_key)

    @retry(wait=wait_exponential(multiplier=2, min=4, max=60), stop=stop_after_attempt(4))
    def review(self, pr_record: dict) -> dict:
        """
        Generate a code review for a single preprocessed PR record.

        Parameters
        ----------
        pr_record : dict with at least 'title' and 'diff' keys

        Returns
        -------
        dict with keys: pr_id, repo, review (the LLM output), model
        """
        user_msg = _USER_TEMPLATE.format(
            title=pr_record.get("title", "Untitled"),
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
            logger.warning(f"PR {pr_record.get('pr_id')}: LLM returned invalid JSON — storing raw text")
            review_json = {"raw": raw}

        return {
            "pr_id": pr_record.get("pr_id"),
            "repo": pr_record.get("repo"),
            "model": self.model,
            "approach": "baseline",
            "review": review_json,
        }

    def review_batch(self, records: list[dict]) -> list[dict]:
        results = []
        for record in records:
            logger.info(f"Baseline review: PR #{record.get('pr_id')}")
            result = self.review(record)
            results.append(result)
        return results
