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
│   └── knowledge_base/      # optional extra .txt/.md rule files for RAG
├── notebooks/
│   └── exploration.ipynb    # interactive data inspection
├── src/
│   ├── data_collection/
│   │   └── github_collector.py   # GitHub API → raw PR JSON
│   ├── preprocessing/
│   │   └── preprocessor.py       # clean diffs, merge comments → normalized JSON
│   ├── baseline/
│   │   └── baseline_model.py     # zero-retrieval LLM reviewer
│   ├── rag/
│   │   ├── document_loader.py    # OWASP + coding rules → LangChain Documents
│   │   ├── vector_store.py       # ChromaDB build / load / query
│   │   └── rag_pipeline.py       # retrieval-augmented LLM reviewer
│   ├── evaluation/
│   │   └── evaluator.py          # semantic sim, ROUGE-L, issue detection, hallucination
│   └── utils.py                  # config loader, logger, env helpers
├── main.py                  # CLI entry point
├── requirements.txt
└── .env.example
```

---

## Setup

### 1. Clone and enter the project
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
```bash
cp .env.example .env
# Edit .env and add your real tokens:
#   GITHUB_TOKEN=ghp_...
#   OPENAI_API_KEY=sk-...
```

### 5. Configure the pipeline
Edit `config/config.yaml` to set:
- `github.repo` — e.g. `"microsoft/vscode"` or a smaller repo
- `github.max_prs` — start with 10–20 for testing
- `llm.model` — `"gpt-4o-mini"` is cheapest; `"gpt-4o"` is more accurate

---

## Running the Pipeline

Each stage can be run independently, or run `all` to execute the full pipeline.

```bash
# Step 1: Collect PR data from GitHub
python main.py collect

# Step 2: Clean and normalize the data
python main.py preprocess

# Step 3: Build the RAG vector store (OWASP + coding rules)
python main.py ingest

# Step 4: Run zero-retrieval LLM reviews
python main.py baseline

# Step 5: Run RAG-augmented LLM reviews
python main.py rag

# Step 6: Evaluate both approaches
python main.py evaluate

# Or run everything at once:
python main.py all
```

---

## Expected Outputs

| File | Description |
|------|-------------|
| `data/raw/prs_raw.json` | Raw PR data from GitHub |
| `data/processed/prs_processed.json` | Normalized `{pr_id, repo, diff, human_review}` records |
| `data/processed/baseline_results.json` | Baseline LLM reviews |
| `data/processed/rag_results.json` | RAG-augmented reviews |
| `data/processed/evaluation_results.json` | Per-PR scores + aggregate summary |
| `data/chroma_db/` | Persisted ChromaDB vector store |

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

---

## Adding Extra Knowledge Base Documents

Drop any `.txt` or `.md` files into `data/knowledge_base/` before running `ingest`.  
They will be chunked and added to the vector store automatically alongside the built-in OWASP Top 10 and secure coding rules.

---

## Tips for a Small Repo

For a quick end-to-end test without burning GitHub API quota, point at a small active repo with many PRs:
```yaml
github:
  repo: "pallets/flask"
  max_prs: 15
```
