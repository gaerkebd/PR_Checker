"""
preliminary_results.py
-----------------------
Generates concrete preliminary results for the midterm progress report.
Run this script before writing Section IV — it saves real numbers to disk.

Stages:
  1. Knowledge-base stats    (no API key needed)
  2. Retrieval smoke test    (needs GEMINI_API_KEY + `python main.py ingest` first)
  3. Mini pilot on 2 sample diffs  (needs GEMINI_API_KEY + ingest)

Output: data/processed/preliminary_results.json
"""

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv
load_dotenv()

_key = os.environ.get("GEMINI_API_KEY", "")
if _key:
    os.environ.setdefault("GOOGLE_API_KEY", _key)

from src.utils import load_config, get_logger
logger = get_logger("preliminary")

config = load_config("config/config.yaml")
OUTPUT_PATH = "data/processed/preliminary_results.json"


# ── Stage 1: Knowledge base stats (no API key required) ──────────────────────

def knowledge_base_stats() -> dict:
    from src.rag.document_loader import load_documents

    docs = load_documents(
        knowledge_base_dir=config["paths"]["knowledge_base"],
        chunk_size=config["embeddings"]["chunk_size"],
        chunk_overlap=config["embeddings"]["chunk_overlap"],
    )

    by_source: dict[str, int] = {}
    for d in docs:
        prefix = d.metadata.get("source", "unknown").split(":")[0].strip()
        by_source[prefix] = by_source.get(prefix, 0) + 1

    print("\n=== Stage 1: Knowledge Base Stats ===")
    print(f"  Total chunks after splitting : {len(docs)}")
    for src, count in sorted(by_source.items()):
        print(f"    {src}: {count} chunks")

    return {"total_chunks": len(docs), "by_source": by_source}


# ── Stage 2: Retrieval smoke test ────────────────────────────────────────────

RETRIEVAL_QUERIES = [
    "SQL injection in user login query",
    "hardcoded API key and password in source code",
    "path traversal via user-supplied filename",
    "JWT token stored in browser localStorage",
]

def retrieval_smoke_test() -> list[dict] | None:
    from src.rag.vector_store import load_vector_store, retrieve

    chroma_path = config["paths"]["chroma_db"]
    if not os.path.exists(chroma_path):
        print("\n[Stage 2 SKIPPED] Vector store not found. Run: python main.py ingest")
        return None

    store = load_vector_store(
        persist_dir=chroma_path,
        collection_name=config["rag"]["collection_name"],
        embedding_model=config["embeddings"]["model"],
    )

    results = []
    print("\n=== Stage 2: Retrieval Smoke Test ===")
    for query in RETRIEVAL_QUERIES:
        docs = retrieve(store, query, top_k=3)
        top = [{"source": d.metadata.get("source", "?"),
                "snippet": d.page_content[:120].strip()} for d in docs]
        print(f"\n  Query: {query!r}")
        for i, t in enumerate(top, 1):
            print(f"    {i}. [{t['source']}]  {t['snippet']}…")
        results.append({"query": query, "top_3": top})

    return results


# ── Stage 3: Mini pilot on 2 synthetic diffs ─────────────────────────────────
# These are small, crafted diffs with known vulnerabilities so results are
# meaningful even without a full GitHub collection run.

SAMPLE_PRS = [
    {
        "pr_id": "pilot_001",
        "repo": "example/webapp",
        "title": "Add user login endpoint",
        "diff": """\
@@ -0,0 +1,18 @@
+def login(username, password):
+    query = "SELECT * FROM users WHERE username='" + username + "' AND password='" + password + "'"
+    result = db.execute(query)
+    if result:
+        secret = 'hardcoded_jwt_secret_123'
+        token = jwt.encode({'user': username}, secret, algorithm='HS256')
+        return {'token': token}
+    return None
""",
        "human_review": [
            "This is vulnerable to SQL injection — use parameterized queries.",
            "The JWT secret is hardcoded. Move it to an environment variable.",
            "No rate limiting on login attempts — brute-force risk.",
        ],
    },
    {
        "pr_id": "pilot_002",
        "repo": "example/api",
        "title": "Add file upload and thumbnail generation",
        "diff": """\
@@ -0,0 +1,14 @@
+import os
+def upload_file(filename, content):
+    # Save uploaded file directly using user-supplied name
+    path = '/uploads/' + filename
+    with open(path, 'wb') as f:
+        f.write(content)
+    # Generate thumbnail
+    os.system('convert ' + path + ' /thumbnails/' + filename)
+    return {'status': 'ok', 'path': path}
""",
        "human_review": [
            "Path traversal — filename is user-controlled and not sanitized.",
            "Command injection in os.system() call with unsanitized filename.",
            "No file type validation — attacker can upload arbitrary files.",
        ],
    },
]

""" FOR API KEY USAGE
def mini_pilot() -> list[dict] | None:
    from src.baseline.baseline_model import BaselineReviewer
    from src.rag.vector_store import load_vector_store
    from src.rag.rag_pipeline import RAGReviewer
    from src.evaluation.evaluator import Evaluator

    #api_key = os.environ.get("GEMINI_API_KEY", "")
    #if not api_key:
    #   print("\n[Stage 3 SKIPPED] GEMINI_API_KEY not set.")
    #    return None




    chroma_path = config["paths"]["chroma_db"]
    if not os.path.exists(chroma_path):
        print("\n[Stage 3 SKIPPED] Vector store not found. Run: python main.py ingest")
        return None

    baseline_reviewer = BaselineReviewer(config)
    store = load_vector_store(
        persist_dir=chroma_path,
        collection_name=config["rag"]["collection_name"],
        embedding_model=config["embeddings"]["model"],
    )
    rag_reviewer = RAGReviewer(config, vector_store=store)
    evaluator = Evaluator(embedding_model=config["embeddings"]["model"])

    pilot_results = []
    print("\n=== Stage 3: Mini Pilot (2 Sample PRs) ===")

    for pr in SAMPLE_PRS:
        print(f"\n  PR: {pr['title']!r}")

        b_out = baseline_reviewer.review(pr)
        r_out = rag_reviewer.review(pr)

        b_scores = evaluator.evaluate_single(b_out, pr)
        r_scores = evaluator.evaluate_single(r_out, pr)

        print(f"    Baseline — issues: {b_scores['n_issues_generated']}, "
              f"sem_sim: {b_scores['semantic_similarity']}, "
              f"issue_det: {b_scores['issue_detection_score']}")
        print(f"    RAG      — issues: {r_scores['n_issues_generated']}, "
              f"sem_sim: {r_scores['semantic_similarity']}, "
              f"issue_det: {r_scores['issue_detection_score']}, "
              f"halluc: {r_scores['hallucination_score']}")

        pilot_results.append({
            "pr_id": pr["pr_id"],
            "title": pr["title"],
            "baseline_scores": b_scores,
            "rag_scores": r_scores,
            "baseline_review": b_out["review"],
            "rag_review": r_out["review"],
        })

    # Print aggregate averages
    def avg(key, results_list):
        vals = [r[key] for r in results_list if r[key] is not None]
        return round(sum(vals) / len(vals), 4) if vals else None

    print("\n  --- Aggregate (2-PR pilot) ---")
    for approach in ("baseline_scores", "rag_scores"):
        label = approach.replace("_scores", "").capitalize()
        for m in ("semantic_similarity", "rouge_l", "issue_detection_score", "hallucination_score"):
            val = avg(m, [r[approach] for r in pilot_results])
            if val is not None:
                print(f"    [{label}] {m}: {val}")

    return pilot_results
"""
# FOR OLLAMA USAGE
def mini_pilot() -> list[dict] | None:
    from src.baseline.baseline_model import BaselineReviewer
    from src.rag.vector_store import load_vector_store
    from src.rag.rag_pipeline import RAGReviewer
    from src.evaluation.evaluator import Evaluator

    ollama_url=config["ollama"]["base_url"]
    model=config["ollama"]["model"]
    stream=config["ollama"]["stream"]

    chroma_path = config["paths"]["chroma_db"]
    if not os.path.exists(chroma_path):
        print("\n[Stage 3 SKIPPED] Vector store not found. Run: python main.py ingest")
        return None

    baseline_reviewer = BaselineReviewer(config)
    store = load_vector_store(
        persist_dir=chroma_path,
        collection_name=config["rag"]["collection_name"],
        embedding_model=config["embeddings"]["model"],
    )
    rag_reviewer = RAGReviewer(config, vector_store=store)
    evaluator = Evaluator(embedding_model=config["embeddings"]["model"])

    pilot_results = []
    print("\n=== Stage 3: Mini Pilot (2 Sample PRs) ===")

    for pr in SAMPLE_PRS:
        print(f"\n  PR: {pr['title']!r}")

        b_out = baseline_reviewer.review(pr)
        r_out = rag_reviewer.review(pr)

        b_scores = evaluator.evaluate_single(b_out, pr)
        r_scores = evaluator.evaluate_single(r_out, pr)

        print(f"    Baseline — issues: {b_scores['n_issues_generated']}, "
              f"sem_sim: {b_scores['semantic_similarity']}, "
              f"issue_det: {b_scores['issue_detection_score']}")
        print(f"    RAG      — issues: {r_scores['n_issues_generated']}, "
              f"sem_sim: {r_scores['semantic_similarity']}, "
              f"issue_det: {r_scores['issue_detection_score']}, "
              f"halluc: {r_scores['hallucination_score']}")

        pilot_results.append({
            "pr_id": pr["pr_id"],
            "title": pr["title"],
            "baseline_scores": b_scores,
            "rag_scores": r_scores,
            "baseline_review": b_out["review"],
            "rag_review": r_out["review"],
        })

    # Print aggregate averages
    def avg(key, results_list):
        vals = [r[key] for r in results_list if r[key] is not None]
        return round(sum(vals) / len(vals), 4) if vals else None

    print("\n  --- Aggregate (2-PR pilot) ---")
    for approach in ("baseline_scores", "rag_scores"):
        label = approach.replace("_scores", "").capitalize()
        for m in ("semantic_similarity", "rouge_l", "issue_detection_score", "hallucination_score"):
            val = avg(m, [r[approach] for r in pilot_results])
            if val is not None:
                print(f"    [{label}] {m}: {val}")

    return pilot_results

# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    results = {}
    results["kb_stats"]             = knowledge_base_stats()
    results["retrieval_smoke_test"] = retrieval_smoke_test()
    results["mini_pilot"]           = mini_pilot()

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\nAll results saved → {OUTPUT_PATH}")



