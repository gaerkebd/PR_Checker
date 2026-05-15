"""
bigvul.py
---------
Downloads the BigVul CVE dataset from HuggingFace, filters for
security-relevant records, and saves them as JSON for downstream
ingestion into the RAG knowledge base.

Dataset: https://huggingface.co/datasets/bstee615/bigvul
"""

import json
import os
import re
import time
from typing import NotRequired, TypedDict

import requests as http_requests
from datasets import load_dataset

from src.utils import get_logger

logger = get_logger(__name__)


# ── Schema ──────────────────────────────────────────────────────────────────

class BigVulRecord(TypedDict):
    """Standardized schema for a single BigVul training example."""
    cve_id: str
    cve_page: str
    cwe_id: str
    codeLink: str
    commit_message: str
    func_after: str
    func_before: str
    lang: str
    cve_description: NotRequired[str]
    cwe_description: NotRequired[str]


# ── CWE filter + descriptions ──────────────────────────────────────────────

_CWE_DESCRIPTIONS: dict[int, str] = {
    # A01 Broken Access Control
    200: "Exposure of Sensitive Information to an Unauthorized Actor.",
    201: "Insertion of Sensitive Information Into Sent Data.",
    284: "Improper Access Control.",
    285: "Improper Authorization.",
    352: "Cross-Site Request Forgery (CSRF): forcing an authenticated user to perform unintended actions.",
    639: "Authorization Bypass Through User-Controlled Key.",
    # A02 Cryptographic Failures
    261: "Weak Encoding for Password.",
    295: "Improper Certificate Validation.",
    296: "Improper Following of a Certificate's Chain of Trust.",
    310: "Cryptographic Issues.",
    319: "Cleartext Transmission of Sensitive Information.",
    326: "Inadequate Encryption Strength.",
    327: "Use of a Broken or Risky Cryptographic Algorithm.",
    328: "Use of Weak Hash.",
    330: "Use of Insufficiently Random Values.",
    # A03 Injection
    20: "Improper Input Validation.",
    77: "Command Injection: improper neutralization of special elements used in a command.",
    78: "OS Command Injection: improper neutralization of special elements used in an OS command.",
    79: "Cross-site Scripting (XSS): improper neutralization of input during web page generation.",
    89: "SQL Injection: improper neutralization of special elements used in an SQL command.",
    94: "Code Injection: improper control of code generation.",
    917: "Expression Language Injection.",
    # A04 Insecure Design
    209: "Information Exposure Through an Error Message.",
    256: "Plaintext Storage of a Password.",
    501: "Trust Boundary Violation.",
    # A05 Security Misconfiguration
    2: "Environment configuration weakness.",
    16: "Incorrect system or application configuration.",
    732: "Incorrect Permission Assignment for Critical Resource.",
    # A06 Vulnerable Components
    937: "Using Components with Known Vulnerabilities.",
    # A07 Auth Failures
    287: "Improper Authentication.",
    306: "Missing Authentication for Critical Function.",
    384: "Session Fixation.",
    613: "Insufficient Session Expiration.",
    # A08 Integrity Failures
    502: "Deserialization of Untrusted Data.",
    829: "Inclusion of Functionality from Untrusted Control Sphere.",
    # A09 Logging Failures
    117: "Improper Output Neutralization for Logs.",
    223: "Omission of Security-relevant Information.",
    532: "Insertion of Sensitive Information into Log File.",
    778: "Insufficient Logging.",
    # A10 SSRF
    918: "Server-Side Request Forgery (SSRF).",
    # Memory safety (very common in BigVul)
    119: "Buffer Overflow: operations on a buffer without proper bounds checking.",
    120: "Classic Buffer Overflow: copying input to a buffer without checking size.",
    121: "Stack-based Buffer Overflow.",
    122: "Heap-based Buffer Overflow.",
    125: "Out-of-bounds Read: reading data past the end of allocated memory.",
    126: "Buffer Over-read.",
    127: "Buffer Under-read.",
    131: "Incorrect Calculation of Buffer Size.",
    190: "Integer Overflow or Wraparound.",
    191: "Integer Underflow.",
    369: "Divide By Zero.",
    399: "Resource Management Errors.",
    400: "Uncontrolled Resource Consumption.",
    401: "Memory Leak: not releasing allocated memory after use.",
    415: "Double Free: freeing the same memory location twice.",
    416: "Use After Free: referencing memory after it has been freed.",
    457: "Use of Uninitialized Variable.",
    476: "NULL Pointer Dereference.",
    667: "Improper Locking.",
    787: "Out-of-bounds Write: writing data past the end or before the beginning of a buffer.",
    824: "Access of Uninitialized Pointer.",
    # Misc high-impact
    362: "Race Condition: concurrent execution leading to unexpected behavior.",
    754: "Improper Check for Unusual or Exceptional Conditions.",
    772: "Missing Release of Resource after Effective Lifetime.",
    835: "Infinite Loop: loop with unreachable exit condition.",
}

_RELEVANT_CWES: set[int] = set(_CWE_DESCRIPTIONS.keys())


# ── Commit-message cleaning ────────────────────────────────────────────────

_TRAILER_RE = re.compile(
    r"^("
    r"Reviewed-by|Signed-off-by|Acked-by|Tested-by|Reported-by|"
    r"Change-Id|Cr-Commit-Position|Commit-Queue|Cq-Include-Trybots|"
    r"Reviewed-on|Cc|Fixes|Link|Message-Id|References|"
    r"Acked-and-tested-by|Suggested-by|Co-developed-by"
    r"):.*$",
    re.MULTILINE | re.IGNORECASE,
)
_BUG_LINE_RE = re.compile(r"^BUG=.*$", re.MULTILINE)


def clean_commit_message(msg: str) -> str:
    """Strip git trailers and metadata noise, keeping the meaningful description."""
    cleaned = _TRAILER_RE.sub("", msg)
    cleaned = _BUG_LINE_RE.sub("", cleaned)
    # Collapse runs of blank lines
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


# ── CWE / CVE helpers ──────────────────────────────────────────────────────

def _parse_cwe_id(raw: str) -> int | None:
    """Normalise CWE strings like 'CWE-79', '79', or '' → int or None."""
    if not raw:
        return None
    cleaned = str(raw).strip().upper().replace("CWE-", "")
    try:
        return int(cleaned)
    except ValueError:
        return None


_NVD_API_URL = "https://services.nvd.nist.gov/rest/json/cves/2.0"


def _fetch_cve_description(cve_id: str, api_key: str | None = None) -> str | None:
    """Fetch the English CVE description from the NVD API v2.0."""
    headers = {}
    if api_key:
        headers["apiKey"] = api_key
    try:
        resp = http_requests.get(
            _NVD_API_URL,
            params={"cveId": cve_id},
            headers=headers,
            timeout=30,
        )
        resp.raise_for_status()
        vulns = resp.json().get("vulnerabilities", [])
        if vulns:
            for desc in vulns[0].get("cve", {}).get("descriptions", []):
                if desc.get("lang") == "en":
                    return desc["value"]
    except Exception as e:
        logger.warning(f"NVD lookup failed for {cve_id}: {e}")
    return None


# ── HuggingFace loader ─────────────────────────────────────────────────────

def load_from_huggingface(
    dataset_id: str = "bstee615/bigvul",
    split: str = "train",
    max_records: int = 500,
) -> list[BigVulRecord]:
    """
    Stream the BigVul dataset from HuggingFace, apply quality + CWE
    filters, deduplicate by CVE, and return up to *max_records* records.
    Commit messages are cleaned during loading.
    """
    logger.info(f"Streaming BigVul dataset ({dataset_id}, split={split}) …")
    ds = load_dataset(dataset_id, split=split, streaming=True)

    records: list[BigVulRecord] = []
    seen_cves: set[str] = set()
    scanned = 0
    skipped_not_vul = 0
    skipped_cwe = 0
    skipped_quality = 0
    skipped_dup = 0

    for row in ds:
        scanned += 1

        # Only keep vulnerable examples
        if not row.get("vul"):
            skipped_not_vul += 1
            continue

        # CWE filter (field name has a space: "CWE ID")
        cwe_int = _parse_cwe_id(str(row.get("CWE ID", "")))
        if cwe_int is None or cwe_int not in _RELEVANT_CWES:
            skipped_cwe += 1
            continue

        # Quality: meaningful commit message + non-empty before-code
        raw_msg = (row.get("commit_message") or "").strip()
        commit_msg = clean_commit_message(raw_msg)
        func_before = (row.get("func_before") or "").strip()
        if len(commit_msg) < 20 or not func_before:
            skipped_quality += 1
            continue

        # Deduplicate by CVE (field name has a space: "CVE ID")
        cve_id = (row.get("CVE ID") or "").strip()
        if cve_id and cve_id in seen_cves:
            skipped_dup += 1
            continue
        if cve_id:
            seen_cves.add(cve_id)

        rec = BigVulRecord(
            cve_id=cve_id,
            cve_page=(row.get("CVE Page") or "").strip(),
            cwe_id=f"CWE-{cwe_int}",
            codeLink=(row.get("codeLink") or "").strip(),
            commit_message=commit_msg,
            func_after=(row.get("func_after") or "").strip(),
            func_before=func_before,
            lang=(row.get("lang") or "").strip(),
        )

        # CWE description is free — add it immediately
        if cwe_int in _CWE_DESCRIPTIONS:
            rec["cwe_description"] = _CWE_DESCRIPTIONS[cwe_int]

        records.append(rec)

        if len(records) >= max_records:
            break

        if scanned % 5000 == 0:
            logger.info(f"  scanned {scanned:,} rows, kept {len(records)} so far …")

    logger.info(
        f"BigVul scan complete: scanned={scanned:,}  kept={len(records)}  "
        f"skipped_not_vul={skipped_not_vul}  skipped_cwe={skipped_cwe}  "
        f"skipped_quality={skipped_quality}  skipped_dup={skipped_dup}"
    )
    return records


# ── CVE enrichment (NVD API) ───────────────────────────────────────────────

def enrich_records(records_path: str) -> None:
    """
    Read saved BigVul records and enrich each with a CVE description
    from the NVD API.  Skips records that already have a description.
    Checkpoints to disk every 10 lookups so progress survives interruption.

    Rate limits:
      - Without NVD_API_KEY env var: ~5 requests / 30 s  (≈ 6 s between calls)
      - With    NVD_API_KEY env var: ~50 requests / 30 s (≈ 0.65 s between calls)
    """
    with open(records_path, encoding="utf-8") as f:
        records: list[dict] = json.load(f)

    api_key = os.environ.get("NVD_API_KEY")
    delay = 0.65 if api_key else 6.5
    if api_key:
        logger.info("NVD_API_KEY found — using fast rate limit (50 req/30 s)")
    else:
        logger.info("No NVD_API_KEY — using public rate limit (5 req/30 s). "
                     "Set NVD_API_KEY in .env for ~10x faster enrichment.")

    pending = [(i, r) for i, r in enumerate(records) if not r.get("cve_description")]
    logger.info(f"Enriching {len(pending)} records (skipping {len(records) - len(pending)} already enriched)")

    fetched = 0
    for count, (i, rec) in enumerate(pending, 1):
        cve_id = rec.get("cve_id", "")
        if not cve_id:
            continue

        desc = _fetch_cve_description(cve_id, api_key=api_key)
        if desc:
            records[i]["cve_description"] = desc
            fetched += 1

        # Checkpoint every 10 lookups
        if count % 10 == 0:
            _save_json(records, records_path)
            logger.info(f"  progress: {count}/{len(pending)} looked up, {fetched} descriptions found")

        time.sleep(delay)

    _save_json(records, records_path)
    logger.info(f"Enrichment complete: {fetched} CVE descriptions added out of {len(pending)} looked up")


# ── Persistence ─────────────────────────────────────────────────────────────

def _save_json(data: list, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def save_records(records: list[BigVulRecord], output_path: str) -> None:
    """Persist BigVul records as JSON."""
    _save_json(records, output_path)
    logger.info(f"Saved {len(records)} BigVul records → {output_path}")
