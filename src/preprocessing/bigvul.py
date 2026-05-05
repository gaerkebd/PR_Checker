"""
bigvul.py
---------
Loads the BigVul vulnerability dataset from HuggingFace and transforms each
row into a standardized record that can be used for training or fine-tuning
a vulnerability-detection model.

Output schema per record
------------------------
{
    "commit_message":   str,  # the commit message associated with the CVE fix
    "commit_diff":      str,  # unified diff / patch introduced by the fix
    "review_comment":   str,  # synthesized review comment from vuln description
    "cve_number":       str,  # e.g. "CVE-2021-1234"
    "cve_reference":    str,  # reference URL / citation for the CVE
    "cve_definition":   str,  # human-readable definition / summary of the vulnerability
}

Usage
-----
    from src.preprocessing.bigvul import load_from_huggingface, save_records

    records = load_from_huggingface(max_records=5000)
    save_records(records, "data/raw/bigvul/bigvul_records.json")
"""

import json
import os
from typing import TypedDict

from src.utils import get_logger

logger = get_logger(__name__)

# HuggingFace dataset identifier for BigVul
BIGVUL_DATASET_ID = "bstee615/bigvul"

# Candidate field names for each concept across known BigVul schema variants.
# The loader tries each name in order and takes the first non-empty hit.
_FIELD_CANDIDATES: dict[str, list[str]] = {
    "commit_message": ["Commit message", "commit_message", "commit_msg", "message"],
    "commit_diff":    ["patch", "diff", "func_diff", "vul_func_with_fix"],
    "review_comment": ["summary", "vulnerability_description", "description", "Abstract"],
    "cve_number":     ["CVE ID", "cve_id", "cve", "CVE"],
    "cve_reference":  ["References", "ref_link", "reference", "ref"],
    "cve_definition": ["summary", "vulnerability_description", "description", "Abstract",
                       "cwe_definition"],
}


class BigVulRecord(TypedDict):
    """
    Standardized schema for a single BigVul training example.

    Fields
    ------
    commit_message : The git commit message that introduced or fixed the vulnerability.
    commit_diff    : The raw unified diff / patch for the commit.
    review_comment : A synthesized code-review style comment derived from the CVE
                     description — acts as the "ground truth" reviewer signal.
    cve_number     : The CVE identifier, e.g. "CVE-2021-44228".
    cve_reference  : A URL or citation pointing to the authoritative CVE reference.
    cve_definition : The full plain-text description / summary of the vulnerability.
    """
    commit_message: str
    commit_diff:    str
    review_comment: str
    cve_number:     str
    cve_reference:  str
    cve_definition: str


# ── Field extraction helpers ─────────────────────────────────────────────────

def _pick(row: dict, candidates: list[str], default: str = "") -> str:
    """Return the first non-empty string value found among candidate field names."""
    for key in candidates:
        val = row.get(key)
        if val and str(val).strip():
            return str(val).strip()
    return default


def _build_review_comment(row: dict) -> str:
    """
    Construct a realistic code-review comment from the vulnerability metadata.

    Combines CVE ID, CWE classification, and the human-readable summary into
    a comment that mirrors what a security-aware reviewer might write.
    """
    summary = _pick(row, ["summary", "vulnerability_description", "Abstract", "description"])
    cve_id  = _pick(row, _FIELD_CANDIDATES["cve_number"])
    cwe_id  = _pick(row, ["CWE ID", "cwe_id", "cwe"])
    cwe_def = _pick(row, ["cwe_definition", "cwe_name"])

    parts: list[str] = []

    if cve_id:
        parts.append(f"Security issue: {cve_id}")

    if cwe_id:
        cwe_line = f"CWE: {cwe_id}"
        if cwe_def:
            cwe_line += f" – {cwe_def}"
        parts.append(cwe_line)

    if summary:
        parts.append(summary)

    return "\n".join(parts) if parts else "Potential vulnerability – see CVE details."


# ── Core transformer ─────────────────────────────────────────────────────────

def transform_row(row: dict) -> BigVulRecord:
    """
    Map one raw BigVul dataset row to a BigVulRecord.

    Parameters
    ----------
    row : raw dict from a HuggingFace datasets iteration (one example)

    Returns
    -------
    BigVulRecord with all six standardized fields populated.
    """
    return BigVulRecord(
        commit_message=_pick(row, _FIELD_CANDIDATES["commit_message"], "(no commit message)"),
        commit_diff   =_pick(row, _FIELD_CANDIDATES["commit_diff"],    "(no diff available)"),
        review_comment=_build_review_comment(row),
        cve_number    =_pick(row, _FIELD_CANDIDATES["cve_number"],     "UNKNOWN"),
        cve_reference =_pick(row, _FIELD_CANDIDATES["cve_reference"],  ""),
        cve_definition=_pick(row, _FIELD_CANDIDATES["cve_definition"], ""),
    )


# ── Loader ───────────────────────────────────────────────────────────────────

def load_from_huggingface(
    dataset_id: str = BIGVUL_DATASET_ID,
    split: str = "train",
    max_records: int | None = None,
) -> list[BigVulRecord]:
    """
    Stream BigVul from HuggingFace and return a list of BigVulRecords.

    Requires the ``datasets`` package:
        pip install datasets

    Parameters
    ----------
    dataset_id   : HuggingFace dataset repo ID (default: bstee615/bigvul)
    split        : dataset split to load  ("train" | "test" | "validation")
    max_records  : optional cap on the number of records to load

    Returns
    -------
    list[BigVulRecord]

    Raises
    ------
    ImportError  : if the ``datasets`` package is not installed
    """
    try:
        from datasets import load_dataset  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "The 'datasets' package is required to load BigVul from HuggingFace.\n"
            "Install it with:  pip install datasets"
        ) from exc

    logger.info(f"Loading BigVul from HuggingFace: {dataset_id!r} (split={split!r})")
    ds = load_dataset(dataset_id, split=split, streaming=True)

    records: list[BigVulRecord] = []
    skipped = 0

    for i, row in enumerate(ds):
        if max_records is not None and i >= max_records:
            break
        try:
            records.append(transform_row(dict(row)))
        except Exception as exc:
            logger.warning(f"Row {i}: skipped — {exc}")
            skipped += 1

    logger.info(f"Loaded {len(records)} BigVul records ({skipped} skipped).")
    return records

# ── Persistence ──────────────────────────────────────────────────────────────

def save_records(records: list[BigVulRecord], path: str) -> None:
    """Persist BigVulRecords to disk as a JSON file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)
    logger.info(f"BigVul records saved → {path}  ({len(records)} records)")
