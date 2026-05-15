"""
baseline_model.py
-----------------
Zero-retrieval LLM reviewer.  Takes a PR diff and returns structured JSON
containing a summary, detected issues, and improvement suggestions.

Supports three backends, selected automatically from config (priority order):
  1. DirectML   (config["directml"] present and torch_directml installed)
  2. Ollama     (config["ollama"] present and api_key omitted)
  3. OpenAI-compatible endpoint  (Gemini, OpenAI, etc.)
"""

import json
import requests

from openai import OpenAI
from tenacity import retry, wait_exponential, stop_after_attempt

from src.utils import get_logger

logger = get_logger(__name__)

try:
    import torch_directml
    _DML_AVAILABLE = True
except ImportError:
    _DML_AVAILABLE = False

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
      "cve": "<number if found>",
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
    """Wraps a DirectML, Ollama, or OpenAI-compatible inference backend."""

    def __init__(self, config: dict, api_key: str = ""):
        llm_cfg = config["llm"]
        ollama_cfg = config.get("ollama")
        dml_cfg = config.get("directml")

        self.temperature = llm_cfg.get("temperature", 0.2)
        self.max_tokens = llm_cfg.get("max_tokens", 1500)

        # Backend priority: DirectML > Ollama > OpenAI-compat
        self._use_directml = bool(dml_cfg) and _DML_AVAILABLE
        self._use_ollama = bool(ollama_cfg) and not api_key and not self._use_directml

        if self._use_directml:
            assert dml_cfg is not None
            self._init_directml(dml_cfg)
        elif self._use_ollama:
            assert ollama_cfg is not None
            self._init_ollama(ollama_cfg)
        else:
            self._init_openai(llm_cfg, api_key)

    # ── Backend initialisation ────────────────────────────────────────────────

    def _init_directml(self, dml_cfg: dict) -> None:
        import torch
        from        rmers import AutoTokenizer, AutoModelForCausalLM  # type: ignore[import-untyped]

        model_id = dml_cfg["model"]
        self.model = model_id
        self._dml_max_diff_chars = dml_cfg.get("max_diff_chars", 3000)
        self._dml_max_new_tokens = dml_cfg.get("max_new_tokens", 1500)

        self._dml_device = torch_directml.device()
        logger.info(f"BaselineReviewer using DirectML device={self._dml_device} model={model_id}")

        self._dml_tokenizer = AutoTokenizer.from_pretrained(model_id)
        self._dml_model = AutoModelForCausalLM.from_pretrained(  # type: ignore[reportCallIssue]
            model_id, torch_dtype=torch.float16
        ).to(self._dml_device)  # type: ignore[reportAttributeAccessIssue]

    def _init_ollama(self, ollama_cfg: dict) -> None:
        self.model = ollama_cfg["model"]
        self._ollama_url = ollama_cfg["base_url"]
        self._ollama_stream = ollama_cfg.get("stream", False)
        self._max_diff_chars = ollama_cfg.get("max_diff_chars", 3000)
        logger.info(f"BaselineReviewer using Ollama: {self._ollama_url} model={self.model}")

    def _init_openai(self, llm_cfg: dict, api_key: str) -> None:
        self.model = llm_cfg["model"]
        base_url = llm_cfg.get("base_url")
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        logger.info(f"BaselineReviewer using OpenAI-compat endpoint: model={self.model}")

    # ── DirectML backend ──────────────────────────────────────────────────────

    def _directml_review(self, pr_record: dict) -> dict:
        import torch
        diff = pr_record.get("diff", "")[:self._dml_max_diff_chars]
        user_msg = _USER_TEMPLATE.format(
            title=pr_record.get("title", "Untitled"),
            diff=diff,
        )
        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ]

        # Use the model's chat template when available
        if hasattr(self._dml_tokenizer, "apply_chat_template"):
            prompt = self._dml_tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            prompt = f"{_SYSTEM_PROMPT}\n\n{user_msg}"

        inputs = self._dml_tokenizer(prompt, return_tensors="pt").to(self._dml_device)

        do_sample = self.temperature > 0
        gen_kwargs = dict(
            max_new_tokens=self._dml_max_new_tokens,
            do_sample=do_sample,
            pad_token_id=self._dml_tokenizer.eos_token_id,
        )
        if do_sample:
            gen_kwargs["temperature"] = self.temperature

        with torch.no_grad():
            output_ids = self._dml_model.generate(**inputs, **gen_kwargs)  # type: ignore[reportAttributeAccessIssue]

        # Decode only the newly generated tokens
        new_ids = output_ids[0][inputs["input_ids"].shape[1]:]
        raw = self._dml_tokenizer.decode(new_ids, skip_special_tokens=True).strip()

        try:
            review_json = json.loads(raw)
        except json.JSONDecodeError:
            logger.warning(f"PR {pr_record.get('pr_id')}: DirectML model returned invalid JSON -- storing raw")
            review_json = {"raw": raw}

        return {
            "pr_id": pr_record.get("pr_id"),
            "repo": pr_record.get("repo"),
            "model": self.model,
            "approach": "baseline",
            "review": review_json,
        }

    # ── Ollama backend ────────────────────────────────────────────────────────

    def _ollama_review(self, pr_record: dict) -> dict:
        diff = pr_record.get("diff", "")[:self._max_diff_chars]
        user_msg = _USER_TEMPLATE.format(
            title=pr_record.get("title", "Untitled"),
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

        raw = response.choices[0].message.content or ""
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
        if self._use_directml:
            return self._directml_review(pr_record)
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
