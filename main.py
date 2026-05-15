"""
main.py
-------
CLI entry point for the PR Review Evaluation system.

Usage
-----
  python main.py collect       # fetch PRs from GitHub
  python main.py preprocess    # clean + normalize raw data
  python main.py bigvul        # fetches + normalizes bigvul records
  python main.py baseline      # run zero-retrieval LLM reviews
  python main.py ingest        # build RAG vector store from knowledge base
  python main.py rag           # run RAG-augmented LLM reviews
  python main.py evaluate      # score both approaches against human reviews
  python main.py all           # run full pipeline end-to-end
"""

import argparse
import json
import os
import sys

from dotenv import load_dotenv

load_dotenv()  # reads .env if present

# Google GenAI SDK and langchain-google-genai both read GOOGLE_API_KEY from env.
# Alias GEMINI_API_KEY → GOOGLE_API_KEY so callers only need one key in .env.
_gemini_key = os.environ.get("GEMINI_API_KEY", "")
if _gemini_key:
    os.environ.setdefault("GOOGLE_API_KEY", _gemini_key)

from src.utils import load_config, require_env, get_logger

logger = get_logger("main")


# ── Path helpers ─────────────────────────────────────────────────────────────

def _patch_repo_paths(config: dict) -> None:
    """
    Overwrite the flat config paths with per-repo sub-directories derived
    from github.repo (e.g. "pytorch/pytorch").

    Layout:
        data/raw/<org>/<repo>/prs_raw.json
        data/processed/<org>/<repo>/prs_processed.json
        data/processed/<org>/<repo>/baseline_results.json
        data/processed/<org>/<repo>/rag_results.json
        data/processed/<org>/<repo>/evaluation_results.json
        data/processed/<org>/<repo>/vulnerability_accuracy.json

    chroma_db and knowledge_base remain global (shared across repos).
    """
    repo = config["github"]["repo"]          # e.g. "pytorch/pytorch"
    raw_dir  = f"data/raw/{repo}"
    proc_dir = f"data/processed/{repo}"

    p = config["paths"]
    p["raw_data"]               = f"{raw_dir}/prs_raw.json"
    p["processed_data"]         = f"{proc_dir}/prs_processed.json"
    p["baseline_results"]       = f"{proc_dir}/baseline_results.json"
    p["rag_results"]            = f"{proc_dir}/rag_results.json"
    p["evaluation_results"]     = f"{proc_dir}/evaluation_results.json"
    p["vulnerability_accuracy"] = f"{proc_dir}/vulnerability_accuracy.json"


# ── Stage functions ──────────────────────────────────────────────────────────

def stage_collect(config: dict) -> None:
    from src.data_collection.github_collector import collect_prs, save_raw

    token = require_env("GITHUB_TOKEN")
    data = collect_prs(config, github_token=token)
    save_raw(data, config["paths"]["raw_data"])


def stage_preprocess(config: dict) -> None:
    from src.preprocessing.preprocessor import load_raw, preprocess, save_processed

    raw = load_raw(config["paths"]["raw_data"])
    processed = preprocess(raw)
    save_processed(processed, config["paths"]["processed_data"])


def stage_bigvul(config: dict) -> None:
    from src.preprocessing.bigvul import load_from_huggingface, save_records, enrich_records

    bigvul_path = config["paths"]["bigvul_data"]

    record_list = load_from_huggingface(
        dataset_id="bstee615/bigvul",
        split="train",
        max_records=config["bigvul"]["max_records"]
    )
    save_records(record_list, bigvul_path)

    # Fetch CVE descriptions from NVD (resumable — safe to interrupt)
    logger.info("Enriching records with CVE descriptions from NVD …")
    enrich_records(bigvul_path)


def stage_ingest(config: dict) -> None:
    from src.rag.document_loader import load_documents
    from src.rag.vector_store import build_vector_store


    bigvul_path = config["paths"].get("bigvul_data")
    if bigvul_path and not os.path.exists(bigvul_path):
        logger.warning(f"BigVul data not found at {bigvul_path}. Run 'bigvul' stage first, or ingest will proceed without it.")
        bigvul_path = None

    docs = load_documents(
        knowledge_base_dir=config["paths"]["knowledge_base"],
        chunk_size=config["ollama"]["chunk_size"],
        chunk_overlap=config["ollama"]["chunk_overlap"],
        bigvul_json_path=bigvul_path,
    )
    build_vector_store(
        documents=docs,
        persist_dir=config["paths"]["chroma_db"],
        collection_name=config["rag"]["collection_name"],
        ollama_cfg=config["ollama"],
    )


def stage_baseline(config: dict) -> None:
    import glob as _glob
    from src.baseline.baseline_model import BaselineReviewer

    reviewer = BaselineReviewer(config, api_key="")  # no API key needed for DirectML or Ollama backends"

    # Discover every prs_processed.json under data/processed/ and run baseline on each.
    processed_files = _glob.glob("data/processed/kubernetes/kubernetes/prs_processed.json", recursive=True)
    if not processed_files:
        logger.error("No prs_processed.json files found under data/processed/. Run 'preprocess' first.")
        sys.exit(1)

    for processed_path in processed_files:
        with open(processed_path, encoding="utf-8") as f:
            records = json.load(f)

        out_path = os.path.join(os.path.dirname(processed_path), "baseline_results.json")

        # Checkpointing: load already-completed results keyed by pr_id
        existing: dict = {}
        if os.path.exists(out_path):
            with open(out_path, encoding="utf-8") as f:
                for r in json.load(f):
                    existing[r["pr_id"]] = r

        pending = [r for r in records if r.get("pr_id") not in existing]

        for record in pending:
            logger.info(f"Baseline review: PR #{record.get('pr_id')} ({record.get('repo', '')})")
            result = reviewer.review(record)
            existing[result["pr_id"]] = result
            logger.info(f"Baseline [{processed_path}]: {len(existing)} done, {len(pending)} remaining")

            # Incremental save after every PR so crashes lose at most one record
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(list(existing.values()), f, indent=2)

        logger.info(f"Baseline results saved → {out_path}")



def stage_rag(config: dict) -> None:
    import glob as _glob
    from src.rag.vector_store import load_vector_store
    from src.rag.rag_pipeline import RAGReviewer


    chroma_path = config["paths"]["chroma_db"]
    if not os.path.exists(chroma_path):
        logger.error(f"Vector store not found at {chroma_path}. Run 'ingest' first.")
        sys.exit(1)

    store = load_vector_store(
        persist_dir=chroma_path,
        collection_name=config["rag"]["collection_name"],
        ollama_cfg=config["ollama"],
    )

    reviewer = RAGReviewer(config, vector_store=store)

    # Discover every prs_processed.json under data/processed/ and run RAG on each.
    processed_files = _glob.glob("data/processed/**/prs_processed.json", recursive=True)
    if not processed_files:
        logger.error("No prs_processed.json files found under data/processed/. Run 'preprocess' first.")
        sys.exit(1)

    for processed_path in processed_files:
        with open(processed_path, encoding="utf-8") as f:
            records = json.load(f)

        out_path = os.path.join(os.path.dirname(processed_path), "rag_results.json")

        # Checkpointing: load already-completed results keyed by pr_id
        existing: dict = {}
        if os.path.exists(out_path):
            with open(out_path, encoding="utf-8") as f:
                for r in json.load(f):
                    existing[r["pr_id"]] = r

        pending = [r for r in records if r.get("pr_id") not in existing]
        logger.info(f"RAG [{processed_path}]: {len(existing)} done, {len(pending)} remaining")

        for record in pending:
            logger.info(f"RAG review: PR #{record.get('pr_id')} ({record.get('repo', '')})")
            result = reviewer.review(record)
            existing[result["pr_id"]] = result
            # Incremental save after every PR so crashes lose at most one record
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(list(existing.values()), f, indent=2)

        logger.info(f"RAG results saved → {out_path}")


def stage_evaluate(config: dict) -> None:
    from src.evaluation.evaluator import Evaluator, save_results

    processed_path = config["paths"]["processed_data"]
    baseline_path = config["paths"]["baseline_results"]
    rag_path = config["paths"]["rag_results"]

    with open(processed_path) as f:
        processed_records = json.load(f)

    all_results = []
    for path in [baseline_path, rag_path]:
        if os.path.exists(path):
            with open(path) as f:
                all_results.extend(json.load(f))
        else:
            logger.warning(f"{path} not found — skipping.")

    if not all_results:
        logger.error("No model results found to evaluate. Run 'baseline' and/or 'rag' first.")
        sys.exit(1)

    evaluator = Evaluator(embedding_model=config["ollama"]["embedding_model"])
    per_pr, summary = evaluator.evaluate_batch(all_results, processed_records)
    save_results(per_pr, summary, config["paths"]["evaluation_results"])

    print("\n=== Evaluation Summary ===")
    for approach, metrics in summary.items():
        print(f"\n[{approach}]")
        for k, v in metrics.items():
            print(f"  {k}: {v}")


def stage_vuln_accuracy(config: dict) -> None:
    from src.evaluation.vulnerabilityaccuracy import run_accuracy_evaluation

    raw_data_dir = "data/raw"                              # scan all repos
    output_report = config["paths"]["vulnerability_accuracy"]

    report = run_accuracy_evaluation(
        raw_data_dir=raw_data_dir,
        output_report=output_report,
    )

    if report:
        summary = report.get("summary", {})
        print("\n=== Vulnerability Accuracy Summary ===")
        for k, v in summary.items():
            print(f"  {k}: {v}")


# ── Main ─────────────────────────────────────────────────────────────────────

STAGES = {
    "collect": stage_collect,
    "preprocess": stage_preprocess,
    "bigvul": stage_bigvul,
    "ingest": stage_ingest,
    "baseline": stage_baseline,
    "rag": stage_rag,
    "evaluate": stage_evaluate,
    "vuln_accuracy": stage_vuln_accuracy,
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="PR Review Evaluation Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "stage",
        choices=[*STAGES.keys(), "all"],
        help="Pipeline stage to run",
    )
    parser.add_argument(
        "--config",
        default="config/config.yaml",
        help="Path to config YAML (default: config/config.yaml)",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    _patch_repo_paths(config)

    if args.stage == "all":
        for name, fn in STAGES.items():
            logger.info(f"=== Running stage: {name} ===")
            fn(config)
    else:
        STAGES[args.stage](config)


if __name__ == "__main__":
    main()
