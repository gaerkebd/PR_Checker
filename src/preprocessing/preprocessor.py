"""
preprocessor.py
---------------
Cleans raw PR data and produces a normalized dataset where human reviewer
comments are merged into a single list under `human_review`.

Output schema per record:
{
    "pr_id":        int,
    "repo":         str,
    "title":        str,
    "diff":         str,          # cleaned unified diff
    "human_review": [str],        # all substantive review comments
}
"""

import json
import os
import re
from typing import Optional

from src.utils import get_logger

logger = get_logger(__name__)

# Diff lines that carry no signal (file metadata, index lines, etc.)
_NOISE_PATTERNS = [
    re.compile(r"^diff --git .*"),
    re.compile(r"^index [0-9a-f]+\.\.[0-9a-f]+ \d+$"),
    re.compile(r"^(---|\+\+\+) (a|b)/"),
    re.compile(r"^Binary files .* differ$"),
]

_MIN_COMMENT_LEN = 10   # ignore trivially short comments like "nit" or "ok"


def _clean_diff(raw_diff: str, max_chars: int = 8000) -> str:
    """
    Remove noise lines from a unified diff and truncate to `max_chars`.

    Keeps:
      - Hunk headers  (@@ … @@)
      - Added lines   (+)
      - Removed lines (-)
      - Context lines (space prefix)
    """
    cleaned_lines = []
    for line in raw_diff.splitlines():
        if any(p.match(line) for p in _NOISE_PATTERNS):
            continue
        cleaned_lines.append(line)

    cleaned = "\n".join(cleaned_lines)
    if len(cleaned) > max_chars:
        cleaned = cleaned[:max_chars] + "\n… [diff truncated]"
    return cleaned


def _extract_human_review(pr_record: dict) -> list[str]:
    """
    Merge review_comments and review bodies into a flat list of strings.
    Filters out empty or trivially short entries.
    """
    comments = []

    for rc in pr_record.get("review_comments", []):
        body = (rc.get("body") or "").strip()
        if len(body) >= _MIN_COMMENT_LEN:
            # Prefix with file path for context
            path = rc.get("path", "")
            entry = f"[{path}] {body}" if path else body
            comments.append(entry)

    for r in pr_record.get("reviews", []):
        body = (r.get("body") or "").strip()
        if len(body) >= _MIN_COMMENT_LEN:
            comments.append(body)

    return comments


def preprocess(raw_records: list[dict]) -> list[dict]:
    """
    Transform raw PR records into normalized dataset records.

    Skips records that have no human review comments after filtering.
    """
    processed = []
    skipped = 0

    for record in raw_records:
        human_review = _extract_human_review(record)

        if not human_review:
            skipped += 1
            logger.debug(f"PR #{record.get('pr_number')}: no substantive comments — skipping")
            continue

        processed.append({
            "pr_id": record["pr_number"],
            "repo": record["repo"],
            "title": record.get("title", ""),
            "diff": _clean_diff(record.get("diff", "")),
            "human_review": human_review,
        })

    logger.info(f"Preprocessed {len(processed)} records (skipped {skipped}).")
    return processed


def load_raw(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_processed(data: list[dict], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    logger.info(f"Processed data saved → {path}")
