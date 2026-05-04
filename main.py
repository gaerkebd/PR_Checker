"""
main.py
-------
CLI entry point for the PR Review Evaluation system.

Usage
-----
  python main.py collect       # fetch PRs from GitHub
  python main.py preprocess    # clean + normalize raw data
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


def stage_ingest(config: dict) -> None:
    from src.rag.document_loader import load_documents
    from src.rag.vector_store import build_vector_store

    docs = load_documents(
        knowledge_base_dir=config["paths"]["knowledge_base"],
        chunk_size=config["embeddings"]["chunk_size"],
        chunk_overlap=config["embeddings"]["chunk_overlap"],
    )
    build_vector_store(
        documents=docs,
        persist_dir=config["paths"]["chroma_db"],
        collection_name=config["rag"]["collection_name"],
        embedding_model=config["embeddings"]["model"],
    )


def stage_baseline(config: dict) -> None:
    from src.baseline.baseline_model import BaselineReviewer

    api_key = require_env("GEMINI_API_KEY")

    processed_path = config["paths"]["processed_data"]
    if not os.path.exists(processed_path):
        logger.error(f"Processed data not found at {processed_path}. Run 'preprocess' first.")
        sys.exit(1)

    with open(processed_path) as f:
        records = json.load(f)

    reviewer = BaselineReviewer(config, api_key=api_key)
    results = reviewer.review_batch(records)

    out_path = config["paths"]["baseline_results"]
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Baseline results saved → {out_path}")


def stage_rag(config: dict) -> None:
    from src.rag.vector_store import load_vector_store
    from src.rag.rag_pipeline import RAGReviewer

    api_key = require_env("GEMINI_API_KEY")

    processed_path = config["paths"]["processed_data"]
    if not os.path.exists(processed_path):
        logger.error(f"Processed data not found at {processed_path}. Run 'preprocess' first.")
        sys.exit(1)

    chroma_path = config["paths"]["chroma_db"]
    if not os.path.exists(chroma_path):
        logger.error(f"Vector store not found at {chroma_path}. Run 'ingest' first.")
        sys.exit(1)

    with open(processed_path) as f:
        records = json.load(f)

    store = load_vector_store(
        persist_dir=chroma_path,
        collection_name=config["rag"]["collection_name"],
        embedding_model=config["embeddings"]["model"],
    )

    reviewer = RAGReviewer(config, vector_store=store, api_key=api_key)
    results = reviewer.review_batch(records)

    out_path = config["paths"]["rag_results"]
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
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

    evaluator = Evaluator(embedding_model=config["embeddings"]["model"])
    per_pr, summary = evaluator.evaluate_batch(all_results, processed_records)
    save_results(per_pr, summary, config["paths"]["evaluation_results"])

    print("\n=== Evaluation Summary ===")
    for approach, metrics in summary.items():
        print(f"\n[{approach}]")
        for k, v in metrics.items():
            print(f"  {k}: {v}")


def stage_vuln_accuracy(config: dict) -> None:
    from src.evaluation.vulnerabilityaccuracy import run_accuracy_evaluation

    raw_data_dir = os.path.dirname(config["paths"]["raw_data"])  # e.g. data/raw
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

    if args.stage == "all":
        for name, fn in STAGES.items():
            logger.info(f"=== Running stage: {name} ===")
            fn(config)
    else:
        STAGES[args.stage](config)


if __name__ == "__main__":
    main()
