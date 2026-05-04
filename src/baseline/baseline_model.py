"""
baseline_model.py
-----------------
Zero-retrieval LLM reviewer.  Takes a PR diff and returns structured JSON
containing a summary, detected issues, and improvement suggestions.

Supports two backends, selected automatically from config:
  - Ollama  (config["ollama"] present and api_key omitted)
  - OpenAI-compatible endpoint  (Gemini, OpenAI, etc.)
"""

import json
import requests

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
    """Wraps either Ollama (/api/generate) or an OpenAI-compatible endpoint."""

    def __init__(self, config: dict, api_key: str = ""):
        llm_cfg = config["llm"]
        ollama_cfg = config.get("ollama")

        # Use Ollama when the config section exists and no API key is supplied
        self._use_ollama = bool(ollama_cfg) and not api_key
        self.temperature = llm_cfg.get("temperature", 0.2)
        self.max_tokens = llm_cfg.get("max_tokens", 1500)

        if self._use_ollama:
            self.model = ollama_cfg["model"]
            self._ollama_url = ollama_cfg["base_url"]
            self._ollama_stream = ollama_cfg.get("stream", False)
            logger.info(f"BaselineReviewer using Ollama: {self._ollama_url} model={self.model}")
        else:
            self.model = llm_cfg["model"]
            base_url = llm_cfg.get("base_url")
            self.client = OpenAI(api_key=api_key, base_url=base_url)
            logger.info(f"BaselineReviewer using OpenAI-compat endpoint: model={self.model}")

    # ── Ollama backend ────────────────────────────────────────────────────────

    def _ollama_review(self, pr_record: dict) -> dict:
        user_msg = _USER_TEMPLATE.format(
            title=pr_record.get("title", "Untitled"),
            diff=pr_record.get("diff", ""),
        )
        # Combine system + user into a single prompt for /api/generate
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
            "approach": "baseline",
            "review": review_json,
        }

    # ── OpenAI-compatible backend ─────────────────────────────────────────────

    def _openai_review(self, pr_record: dict) -> dict:
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
            logger.warning(f"PR {pr_record.get('pr_id')}: LLM returned invalid JSON -- storing raw text")
            review_json = {"raw": raw}

        return {
            "pr_id": pr_record.get("pr_id"),
            "repo": pr_record.get("repo"),
            "model": self.model,
            "approach": "baseline",
            "review": review_json,
        }

    # ── Public interface ──────────────────────────────────────────────────────

    @retry(wait=wait_exponential(multiplier=2, min=4, max=60), stop=stop_after_attempt(4))
    def review(self, pr_record: dict) -> dict:
        if self._use_ollama:
            return self._ollama_review(pr_record)
        return self._openai_review(pr_record)

    def review_batch(self, records: list[dict]) -> list[dict]:
        results = []
        for record in records:
            logger.info(f"Baseline review: PR #{record.get('pr_id')}")
            result = self.review(record)
            results.append(result)
        return results
