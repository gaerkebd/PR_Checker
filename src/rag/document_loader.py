"""
document_loader.py
------------------
Loads security / coding-rule documents (plain text or markdown files),
chunks them, and returns LangChain Document objects ready for embedding.

Drop files into data/knowledge_base/ and they are picked up automatically.
Also includes built-in OWASP Top 10 summaries so the system works out of the
box without requiring external files.
"""

import json
import os
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from src.utils import get_logger

logger = get_logger(__name__)

# ── Built-in OWASP Top 10 (2021) summaries ─────────────────────────────────

_OWASP_DOCS = [
    ("A01 Broken Access Control",
     "Broken Access Control moves up from the fifth position; 94% of applications were tested for some form of broken access control. "
     "Notable CWEs: CWE-200 Exposure of Sensitive Information, CWE-201, CWE-352 CSRF. "
     "Prevention: deny by default, enforce record ownership, disable directory listing, log failures, rate-limit API."),

    ("A02 Cryptographic Failures",
     "Formerly 'Sensitive Data Exposure'. Focus on failures related to cryptography that expose sensitive data. "
     "Prevention: classify data, do not store sensitive data unnecessarily, encrypt at rest and in transit, "
     "use strong modern algorithms (AES-256, RSA-2048+, TLS 1.2+), never use MD5/SHA1 for passwords."),

    ("A03 Injection",
     "SQL, NoSQL, OS, LDAP injection. Hostile data is sent to an interpreter as part of a command or query. "
     "Prevention: use parameterized queries / prepared statements, ORMs, input validation, LIMIT query results."),

    ("A04 Insecure Design",
     "Focuses on design and architectural flaws. Missing or ineffective control design. "
     "Prevention: threat modelling, secure design patterns, reference architectures, unit & integration tests for critical flows."),

    ("A05 Security Misconfiguration",
     "90% of apps tested. Includes unnecessary features enabled, default accounts, verbose error messages, "
     "missing security hardening, insecure cloud storage. Prevention: repeatable hardening process, minimal platform, "
     "review cloud storage permissions, segmented architecture, automated config verification."),

    ("A06 Vulnerable and Outdated Components",
     "Using components with known vulnerabilities. Prevention: remove unused dependencies, monitor CVEs, "
     "use SCA tools (OWASP Dependency-Check, Snyk), subscribe to security bulletins, patch promptly."),

    ("A07 Identification and Authentication Failures",
     "Broken authentication. Prevention: MFA, no default credentials, implement brute-force protection, "
     "use server-side session management, invalidate sessions on logout, weak password checks."),

    ("A08 Software and Data Integrity Failures",
     "Code and infrastructure that does not protect against integrity violations, insecure deserialization, "
     "CI/CD pipeline compromise. Prevention: digital signatures, dependency integrity checks, code review of CI/CD, "
     "do not deserialize from untrusted sources."),

    ("A09 Security Logging and Monitoring Failures",
     "Insufficient logging allows attackers to stay undetected. Prevention: log authentication, access control, "
     "input validation failures; ensure logs contain enough context; monitor and alert anomalies; "
     "use centralized log management (SIEM)."),

    ("A10 Server-Side Request Forgery (SSRF)",
     "SSRF occurs when a web app fetches a remote resource without validating the user-supplied URL. "
     "Prevention: sanitize and validate all client-supplied input data, enforce URL schema, port, and destination with allow list, "
     "disable HTTP redirections, do not send raw responses to clients."),
]

# ── Secure coding rules ──────────────────────────────────────────────────────

_CODING_RULES = [
    ("Input Validation",
     "Always validate and sanitize inputs at system boundaries. Use allowlists, not denylists. "
     "Reject unexpected characters early. Never trust client-supplied data."),

    ("Secret Management",
     "Never hard-code secrets, API keys, or passwords in source code. Use environment variables or secret managers. "
     "Secrets committed to git are considered compromised."),

    ("Error Handling",
     "Do not expose stack traces or internal details to end users. Log errors server-side with enough context for debugging. "
     "Use generic error messages for authentication failures."),

    ("SQL Safety",
     "Always use parameterized queries or prepared statements. Never concatenate user input into SQL strings. "
     "Use an ORM where possible and treat ORM output as trusted only after schema validation."),

    ("Dependency Safety",
     "Pin dependency versions. Audit new dependencies before adding them. "
     "Regularly run dependency vulnerability scans. Remove unused packages."),

    ("Authentication Tokens",
     "Use short-lived JWTs. Validate signature, expiry, issuer, and audience. "
     "Store tokens in HttpOnly cookies, not localStorage. Rotate refresh tokens on use."),

    ("Sensitive Data Exposure",
     "Do not log passwords, tokens, PII, or payment info. Mask sensitive fields in logs. "
     "Encrypt sensitive data at rest with AES-256 or equivalent."),

    ("Race Conditions and TOCTOU",
     "Avoid time-of-check to time-of-use bugs. Use atomic operations or database transactions "
     "to ensure state is consistent between check and use."),
]


def _make_documents(pairs: list[tuple[str, str]], source_prefix: str) -> list[Document]:
    return [
        Document(page_content=content, metadata={"source": f"{source_prefix}: {title}"})
        for title, content in pairs
    ]


def _load_bigvul_documents(bigvul_json_path: str) -> list[Document]:
    """
    Read BigVul records from JSON and convert each into a LangChain Document.
    Returns an empty list if the file does not exist.
    """
    if not os.path.exists(bigvul_json_path):
        return []

    with open(bigvul_json_path, encoding="utf-8") as f:
        records = json.load(f)

    docs: list[Document] = []
    for rec in records:
        # Build description block from available fields
        header = f"Security Vulnerability Example: {rec.get('cve_id', '')} ({rec.get('cwe_id', '')})"

        parts = [header]

        cwe_desc = rec.get("cwe_description", "")
        if cwe_desc:
            parts.append(f"Vulnerability Type: {cwe_desc}")

        cve_desc = rec.get("cve_description", "")
        if cve_desc:
            parts.append(f"Description: {cve_desc}")

        parts.append(f"Commit: {rec.get('commit_message', '')}")

        func_before = rec.get("func_before", "")[:500]
        func_after = rec.get("func_after", "")[:500]
        lang = rec.get("lang", "")
        if func_before:
            parts.append(f"Vulnerable code ({lang}):\n{func_before}")
        if func_after:
            parts.append(f"Fixed code:\n{func_after}")

        content = "\n\n".join(parts)

        docs.append(Document(
            page_content=content,
            metadata={
                "source": f"BigVul: {rec.get('cve_id', '')} ({rec.get('cwe_id', '')})",
                "cve_id": rec.get("cve_id", ""),
                "cwe_id": rec.get("cwe_id", ""),
            },
        ))
    return docs


def load_documents(knowledge_base_dir: str, chunk_size: int = 500, chunk_overlap: int = 50,
                   bigvul_json_path: str | None = None) -> list[Document]:
    """
    Load all documents for the RAG knowledge base.

    1. Built-in OWASP Top 10 + secure coding rules (always included).
    2. Any .txt or .md files found in `knowledge_base_dir` (optional extras).

    Returns chunked LangChain Document objects.
    """
    docs: list[Document] = []

    # Built-ins
    docs.extend(_make_documents(_OWASP_DOCS, "OWASP Top 10 2021"))
    docs.extend(_make_documents(_CODING_RULES, "Secure Coding Rule"))
    logger.info(f"Loaded {len(docs)} built-in knowledge documents.")

    # User-supplied files
    kb_path = Path(knowledge_base_dir)
    if kb_path.exists():
        user_files = list(kb_path.glob("**/*.txt")) + list(kb_path.glob("**/*.md"))
        for fpath in user_files:
            text = fpath.read_text(encoding="utf-8", errors="ignore")
            docs.append(Document(page_content=text, metadata={"source": str(fpath)}))
            logger.info(f"Loaded external doc: {fpath}")

    # BigVul CVE examples
    if bigvul_json_path:
        bigvul_docs = _load_bigvul_documents(bigvul_json_path)
        docs.extend(bigvul_docs)
        logger.info(f"Loaded {len(bigvul_docs)} BigVul CVE documents.")

    # Chunk everything
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " "],
    )
    chunked = splitter.split_documents(docs)
    logger.info(f"Total chunks after splitting: {len(chunked)}")
    return chunked
