"""
document_loader.py
------------------
Loads security documents for the RAG knowledge base:
  1. OWASP Top 10 and any other .txt / .md files in knowledge_base_dir
  2. BigVul CVE examples from JSON

BigVul records are embedded as-is (one document per record, no re-chunking).
The semantic anchor is description text; code snippets are truncated so the
total fits well within nomic-embed-text's effective range (~300-400 tokens).

OWASP and other text files use a section-aware splitter tuned for the
slide-format paragraph structure of the OWASP document.
"""

import json
import os
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from src.utils import get_logger

logger = get_logger(__name__)

# Truncation limits for BigVul record fields (chars, not tokens)
_CODE_TRUNCATE = 350   # per code snippet (vulnerable / fixed)
_MAX_CVE_DESC  = 500   # CVE description
_MAX_COMMIT    = 200   # commit message


def _load_bigvul_documents(bigvul_json_path: str) -> list[Document]:
    """
    Convert each BigVul record into one LangChain Document.

    page_content — semantic text for embedding:
        [CWE-X] <cwe_description>
        CVE: <cve_description (truncated)>
        Fix: <commit_message (truncated)>
        Vulnerable (<lang>): <func_before (truncated)>
        Fixed: <func_after (truncated)>

    metadata — stored in ChromaDB for filtering / display:
        source, cve_id, cwe_id, lang, cve_page, commit
    """
    if not os.path.exists(bigvul_json_path):
        logger.warning(f"BigVul JSON not found: {bigvul_json_path}")
        return []

    with open(bigvul_json_path, encoding="utf-8") as f:
        records = json.load(f)

    docs: list[Document] = []
    for rec in records:
        cve_id   = rec.get("cve_id", "")
        cwe_id   = rec.get("cwe_id", "")
        lang     = rec.get("lang", "")
        cve_page = rec.get("cve_page", "")

        cwe_desc    = rec.get("cwe_description", "")
        cve_desc    = (rec.get("cve_description") or "")[:_MAX_CVE_DESC]
        commit      = (rec.get("commit_message")  or "")[:_MAX_COMMIT]
        code_before = (rec.get("func_before")     or "")[:_CODE_TRUNCATE]
        code_after  = (rec.get("func_after")      or "")[:_CODE_TRUNCATE]

        parts: list[str] = []
        if cwe_id and cwe_desc:
            parts.append(f"[{cwe_id}] {cwe_desc}")
        elif cwe_id:
            parts.append(f"[{cwe_id}]")
        if cve_desc:
            parts.append(f"{cve_id}: {cve_desc}")
        if commit:
            parts.append(f"Fix: {commit}")
        if code_before:
            parts.append(f"Vulnerable ({lang}):\n{code_before}")
        if code_after:
            parts.append(f"Fixed:\n{code_after}")

        page_content = "\n\n".join(parts)

        docs.append(Document(
            page_content=page_content,
            metadata={
                "source":   f"BigVul:{cve_id}",
                "cve_id":   cve_id,
                "cwe_id":   cwe_id,
                "lang":     lang,
                "cve_page": cve_page,
                "commit":   commit,
            },
        ))

    logger.info(f"Loaded {len(docs)} BigVul documents from {bigvul_json_path}")
    return docs


def load_documents(
    knowledge_base_dir: str,
    chunk_size: int = 450,
    chunk_overlap: int = 90,
    bigvul_json_path: str | None = None,
) -> list[Document]:
    """
    Load all RAG knowledge-base documents.

    Text files (OWASP, .txt, .md):
      Chunked with a section-aware splitter. Separators prioritise the
      'A0X:2021' OWASP category headers, then blank lines, then newlines.
      chunk_size=450 keeps each chunk to one focused concept (~110 tokens).

    BigVul records:
      One document per record, no re-chunking. Each is already ~300-400
      tokens — within nomic-embed-text's effective embedding range.
    """
    text_docs:   list[Document] = []
    bigvul_docs: list[Document] = []

    # ── Text / OWASP files ─────────────────────────────────────────────────
    kb_path = Path(knowledge_base_dir)
    if kb_path.exists():
        text_files = (
            list(kb_path.glob("**/*.txt")) +
            list(kb_path.glob("**/*.md"))
        )
        for fpath in text_files:
            text = fpath.read_text(encoding="utf-8", errors="ignore")
            text_docs.append(Document(
                page_content=text,
                metadata={"source": fpath.name},
            ))
            logger.info(f"Loaded text doc: {fpath.name} ({len(text):,} chars)")
    else:
        logger.warning(f"knowledge_base_dir not found: {knowledge_base_dir}")

    # Section-aware splitter: respects OWASP A0X category headers before
    # falling back to blank lines and then individual newlines.
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\nA0", "\n\n", "\n", ". ", " "],
    )
    chunked_text = text_splitter.split_documents(text_docs)
    logger.info(
        f"Text docs → {len(chunked_text)} chunks "
        f"(chunk_size={chunk_size}, overlap={chunk_overlap})"
    )

    # ── BigVul CVE records ─────────────────────────────────────────────────
    if bigvul_json_path:
        bigvul_docs = _load_bigvul_documents(bigvul_json_path)

    all_docs = chunked_text + bigvul_docs
    logger.info(
        f"Total documents for embedding: {len(all_docs)} "
        f"({len(chunked_text)} text chunks + {len(bigvul_docs)} BigVul records)"
    )
    return all_docs
