# PR Review Evaluation System

Compares **Baseline LLM** vs **RAG-augmented LLM** for automated GitHub PR code review.

---

## Project Structure

```
PR_Checker/
├── config/
│   └── config.yaml          # all tuneable settings (repo, model, paths…)
├── data/
│   ├── raw/                 # JSON output from GitHub collection (git-ignored)
│   ├── processed/           # cleaned data + model outputs (git-ignored)
│   ├── chroma_db/           # persisted ChromaDB vector store (git-ignored)
│   └── knowledge_base/      # extra .txt/.md rule files for RAG
├── notebooks/
│   └── exploration.ipynb    # interactive data inspection
├── src/
│   ├── data_collection/
│   │   └── github_collector.py        # GitHub API → raw PR JSON
│   ├── preprocessing/
│   │   ├── preprocessor.py            # clean diffs, merge comments → normalized JSON
│   │   └── bigvul.py                  # BigVul CVE dataset loader (HuggingFace)
│   ├── baseline/
│   │   └── baseline_model.py          # zero-retrieval LLM reviewer
│   ├── rag/
│   │   ├── document_loader.py         # OWASP + coding rules → LangChain Documents
│   │   ├── vector_store.py            # ChromaDB build / load / query
│   │   └── rag_pipeline.py            # retrieval-augmented LLM reviewer
│   ├── evaluation/
│   │   ├── evaluator.py               # semantic sim, ROUGE-L, issue detection, hallucination
│   │   └── vulnerabilityaccuracy.py   # keyword-based vuln detection rate across repos
│   └── utils.py                       # config loader, logger, env helpers
├── main.py                  # CLI entry point
├── requirements.txt
└── .env
```

---

## Setup

### 1. Enter the project directory
```bash
cd PR_Checker
```

### 2. Create and activate a virtual environment
```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure secrets
Edit `.env` and set your tokens:
```
GITHUB_TOKEN=ghp_...
GEMINI_API_KEY=AIza...
```

`GEMINI_API_KEY` is used for both the LLM (`gemini-2.0-flash-lite` via the OpenAI-compatible endpoint) and embeddings (`text-embedding-004`). Get a free key at [aistudio.google.com](https://aistudio.google.com).

**Free tier limits** (no billing required):
| Resource | Limit |
|---|---|
| `gemini-2.0-flash-lite` | 30 RPM · 1 500 RPD |
| `text-embedding-004` | 100 RPM · 1 500 RPD |

### 5. Configure the target repo
Edit `config/config.yaml`:
```yaml
github:
  repo: "owner/repo"   # e.g. "pallets/flask" for a quick test
  max_prs: 50          # start small; each PR costs ~1 LLM call per stage
```

---

## Running the Pipeline

Each stage is independent. Run them in order, or use `all` to execute the full pipeline.

```bash
# Step 1 — Fetch closed PRs with review comments from GitHub
python main.py collect
#   writes → data/raw/prs_raw.json

# Step 2 — Clean diffs and normalise human review comments
python main.py preprocess
#   reads  ← data/raw/prs_raw.json
#   writes → data/processed/prs_processed.json

# Step 3 — Embed OWASP + coding rules into ChromaDB
python main.py ingest
#   writes → data/chroma_db/

# Step 4 — Zero-retrieval LLM review (Gemini, no context injection)
python main.py baseline
#   reads  ← data/processed/prs_processed.json
#   writes → data/processed/baseline_results.json

# Step 5 — RAG-augmented LLM review (retrieves top-5 rules per diff)
python main.py rag
#   reads  ← data/processed/prs_processed.json + data/chroma_db/
#   writes → data/processed/rag_results.json

# Step 6 — Score both approaches against human reviewer comments
python main.py evaluate
#   reads  ← data/processed/prs_processed.json
#            data/processed/baseline_results.json
#            data/processed/rag_results.json
#   writes → data/processed/evaluation_results.json

# Step 7 (optional) — Measure human vuln-detection rate across raw PR batches
python main.py vuln_accuracy
#   reads  ← data/raw/   (all prs_raw.json files found recursively)
#   writes → data/processed/vulnerability_accuracy.json

# Run all stages end-to-end
python main.py all
```

Pass `--config path/to/other.yaml` to use a non-default config file.

---

## Output Files

| File | Stage | Description |
|---|---|---|
| `data/raw/prs_raw.json` | collect | Raw PR records: diff, review comments, review bodies |
| `data/processed/prs_processed.json` | preprocess | Normalised `{pr_id, repo, title, diff, human_review}` |
| `data/chroma_db/` | ingest | ChromaDB vector store (OWASP + coding rules, chunked) |
| `data/processed/baseline_results.json` | baseline | Gemini reviews with no retrieved context |
| `data/processed/rag_results.json` | rag | Gemini reviews augmented with top-5 rule chunks |
| `data/processed/evaluation_results.json` | evaluate | Per-PR scores + aggregate summary |
| `data/processed/vulnerability_accuracy.json` | vuln_accuracy | Vuln-detection rate per repo |

### Sample evaluation output
```json
{
  "summary": {
    "baseline": {
      "semantic_similarity": 0.61,
      "rouge_l": 0.12,
      "issue_detection_score": 0.38,
      "hallucination_score": null
    },
    "rag": {
      "semantic_similarity": 0.68,
      "rouge_l": 0.15,
      "issue_detection_score": 0.45,
      "hallucination_score": 0.87
    }
  }
}
```

**Metrics explained:**
- `semantic_similarity` — cosine similarity between generated review text and human comments (Google `text-embedding-004`)
- `rouge_l` — ROUGE-L F1 n-gram overlap
- `issue_detection_score` — keyword overlap between generated issues/suggestions and human comments
- `hallucination_score` — RAG only: fraction of cited rule names that were actually in the retrieved chunks (1.0 = no hallucination)

---

## Adding Extra Knowledge Base Documents

Drop any `.txt` or `.md` files into `data/knowledge_base/` before running `ingest`.  
They are chunked and embedded into ChromaDB automatically alongside the built-in OWASP Top 10 and secure coding rules.

---

## Tips for a Small Test Run

To verify the full pipeline works without burning rate-limit quota:
```yaml
# config/config.yaml
github:
  repo: "pallets/flask"
  max_prs: 10
```

Then run each stage in order. At 10 PRs the full pipeline completes in a few minutes.

---
  