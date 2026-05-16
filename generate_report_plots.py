"""
generate_report_plots.py
------------------------
Generates all figures for the CSE 434 project report.

Figures saved to figures/:
  fig1_vuln_detection_rate.png   – detection rate per repo (bar)
  fig2_metric_comparison.png     – baseline vs RAG grouped bar
  fig3_distributions.png         – per-PR violin plots
  fig4_scatter.png               – ROUGE-L vs semantic similarity
  fig5_hallucination.png         – RAG hallucination score histogram
  fig6_issue_counts.png          – generated vs human issue counts (scatter)
  fig7_qualitative_table.png     – side-by-side qualitative example table
  fig8_per_repo_heatmap.png      – metric heatmap across repos

Usage:
  python generate_report_plots.py
"""

import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import numpy as np

# ── Style ─────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":      "DejaVu Sans",
    "font.size":        11,
    "axes.titlesize":   13,
    "axes.labelsize":   11,
    "legend.fontsize":  10,
    "figure.dpi":       150,
    "axes.spines.top":  False,
    "axes.spines.right": False,
    "axes.grid":        True,
    "grid.alpha":       0.3,
    "grid.linestyle":   "--",
})

BASELINE_COLOR = "#4C72B0"
RAG_COLOR      = "#DD8452"
ACCENT_RED     = "#C44E52"
FIGURES_DIR    = Path("figures")
FIGURES_DIR.mkdir(exist_ok=True)

# ── Data loaders ──────────────────────────────────────────────────────────────

def load_vuln_accuracy(path="data/processed/facebook/react/vulnerability_accuracy.json"):
    with open(path) as f:
        return json.load(f)

def load_eval_results():
    """Return list of (repo_label, per_pr_list) for every evaluation_results.json found."""
    results = []
    for p in sorted(Path("data/processed").rglob("evaluation_results.json")):
        with open(p) as f:
            data = json.load(f)
        label = "/".join(p.parts[2:-1])   # e.g. "django/django" or "openai-agents"
        results.append((label, data))
    return results

def load_baseline_results():
    """Return list of (repo_label, records) for every baseline_results.json found."""
    results = []
    for p in sorted(Path("data/processed").rglob("baseline_results.json")):
        with open(p) as f:
            data = json.load(f)
        label = "/".join(p.parts[2:-1])
        results.append((label, data))
    return results


# ── Figure 1: Vulnerability Detection Rate ────────────────────────────────────

def fig1_vuln_detection_rate(vuln_data):
    per_repo = vuln_data["per_repo"]
    repos    = [r["repo"] for r in per_repo]
    rates    = [r["detection_rate"] * 100 for r in per_repo]
    overall  = vuln_data["summary"]["overall_detection_rate"] * 100

    fig, ax = plt.subplots(figsize=(9, 5))
    colors = [ACCENT_RED if r < overall else BASELINE_COLOR for r in rates]
    bars = ax.barh(repos, rates, color=colors, edgecolor="white", height=0.6)

    # Annotate bars
    for bar, rate in zip(bars, rates):
        ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
                f"{rate:.1f}%", va="center", fontsize=10)

    # Overall average line
    ax.axvline(overall, color="black", linestyle="--", linewidth=1.4, label=f"Overall avg {overall:.1f}%")
    ax.set_xlabel("Human Reviewer Security Detection Rate (%)")
    ax.set_title("Fig 1 — Vulnerability Signal Detection Rate per Repository\n"
                 "(fraction of PRs where ≥1 human reviewer mentioned a security topic)")
    ax.set_xlim(0, max(rates) * 1.25)
    ax.legend()

    above = mpatches.Patch(color=BASELINE_COLOR, label="Above average")
    below = mpatches.Patch(color=ACCENT_RED,     label="Below average")
    ax.legend(handles=[above, below,
                        mlines.Line2D([0], [0], color="black", linestyle="--", label=f"Overall avg {overall:.1f}%")],
              loc="lower right", framealpha=0.85)

    fig.tight_layout()
    out = FIGURES_DIR / "fig1_vuln_detection_rate.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


# ── Figure 2: Baseline vs RAG metric comparison ───────────────────────────────

def fig2_metric_comparison(eval_results):
    """Grouped bar chart: baseline vs RAG for each metric, per repo."""
    metrics = ["semantic_similarity", "rouge_l", "issue_detection_score", "hallucination_score"]
    labels  = ["Semantic\nSimilarity", "ROUGE-L", "Issue\nDetection", "Hallucination\nScore"]

    n_repos = len(eval_results)
    fig, axes = plt.subplots(1, n_repos, figsize=(7 * n_repos, 5), sharey=False)
    if n_repos == 1:
        axes = [axes]

    for ax, (repo, data) in zip(axes, eval_results):
        summary = data["summary"]
        baseline_vals = [summary.get("baseline", {}).get(m, 0) for m in metrics]
        rag_vals      = [summary.get("rag",      {}).get(m, 0) for m in metrics]

        x = np.arange(len(metrics))
        w = 0.35
        b_bars = ax.bar(x - w/2, baseline_vals, w, label="Baseline", color=BASELINE_COLOR, edgecolor="white")
        r_bars = ax.bar(x + w/2, rag_vals,      w, label="RAG",      color=RAG_COLOR,      edgecolor="white")

        # Value labels
        for bar in list(b_bars) + list(r_bars):
            h = bar.get_height()
            if h != 0:
                ax.text(bar.get_x() + bar.get_width() / 2, h + 0.008,
                        f"{h:.3f}", ha="center", va="bottom", fontsize=8)

        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylim(0, 1.1)
        ax.set_ylabel("Score")
        ax.set_title(f"Fig 2 — Baseline vs RAG\n[{repo}]")
        ax.legend()

        # Annotate hallucination note
        ax.annotate("* −1.0 = N/A\n  (baseline has no\n  retrieved sources)",
                    xy=(3 - w/2, baseline_vals[3]),
                    xytext=(2.0, 0.6),
                    fontsize=7.5,
                    arrowprops=dict(arrowstyle="->", color="grey"),
                    color="grey")

    fig.suptitle("Baseline vs RAG: Average Metric Scores per Repository", fontsize=14, y=1.02)
    fig.tight_layout()
    out = FIGURES_DIR / "fig2_metric_comparison.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


# ── Figure 3: Per-PR distributions (violin) ───────────────────────────────────

def fig3_distributions(eval_results):
    """Violin plot comparing per-PR distributions of key metrics."""
    metrics = ["semantic_similarity", "rouge_l", "issue_detection_score"]
    titles  = ["Semantic Similarity", "ROUGE-L F1", "Issue Detection Score"]

    # Gather all per_pr rows across all repos
    all_rows = []
    for _, data in eval_results:
        all_rows.extend(data["per_pr"])

    baseline_data = [r for r in all_rows if r["approach"] == "baseline"]
    rag_data      = [r for r in all_rows if r["approach"] == "rag"]

    fig, axes = plt.subplots(1, len(metrics), figsize=(14, 5))

    for ax, metric, title in zip(axes, metrics, titles):
        bvals = [r[metric] for r in baseline_data if r.get(metric) is not None]
        rvals = [r[metric] for r in rag_data      if r.get(metric) is not None]

        parts = ax.violinplot([bvals, rvals], positions=[1, 2],
                               showmedians=True, showextrema=True)
        for pc, color in zip(parts["bodies"], [BASELINE_COLOR, RAG_COLOR]):
            pc.set_facecolor(color)
            pc.set_alpha(0.7)
        parts["cmedians"].set_color("black")
        parts["cmins"].set_color("grey")
        parts["cmaxes"].set_color("grey")
        parts["cbars"].set_color("grey")

        ax.set_xticks([1, 2])
        ax.set_xticklabels(["Baseline", "RAG"])
        ax.set_ylabel(title)
        ax.set_title(f"{title}\nBaseline μ={np.mean(bvals):.3f}  RAG μ={np.mean(rvals):.3f}")

        # Overlay individual points (jittered)
        for pos, vals, color in [(1, bvals, BASELINE_COLOR), (2, rvals, RAG_COLOR)]:
            jitter = np.random.default_rng(42).uniform(-0.08, 0.08, size=len(vals))
            ax.scatter(np.full(len(vals), pos) + jitter, vals,
                       alpha=0.25, s=8, color=color, zorder=3)

    fig.suptitle("Fig 3 — Per-PR Score Distributions: Baseline vs RAG\n"
                 "(all evaluated repositories combined)", fontsize=13)
    fig.tight_layout()
    out = FIGURES_DIR / "fig3_distributions.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


# ── Figure 4: Scatter — ROUGE-L vs Semantic Similarity ───────────────────────

def fig4_scatter(eval_results):
    all_rows = []
    for repo, data in eval_results:
        for r in data["per_pr"]:
            r["_repo"] = repo
            all_rows.append(r)

    fig, axes = plt.subplots(1, len(eval_results), figsize=(7 * len(eval_results), 5))
    if len(eval_results) == 1:
        axes = [axes]

    for ax, (repo, data) in zip(axes, eval_results):
        for approach, color, marker in [("baseline", BASELINE_COLOR, "o"), ("rag", RAG_COLOR, "^")]:
            pts = [r for r in data["per_pr"] if r["approach"] == approach]
            xs  = [r["rouge_l"] for r in pts]
            ys  = [r["semantic_similarity"] for r in pts]
            ax.scatter(xs, ys, c=color, marker=marker, alpha=0.6, s=40,
                       label=approach.capitalize(), edgecolors="white", linewidths=0.4)

            # Trend line
            if len(xs) > 2:
                z = np.polyfit(xs, ys, 1)
                p = np.poly1d(z)
                xline = np.linspace(min(xs), max(xs), 50)
                ax.plot(xline, p(xline), color=color, linewidth=1.2, linestyle="--", alpha=0.8)

        ax.set_xlabel("ROUGE-L F1")
        ax.set_ylabel("Semantic Similarity")
        ax.set_title(f"Fig 4 — ROUGE-L vs Semantic Similarity\n[{repo}]")
        ax.legend()

    fig.suptitle("ROUGE-L vs Semantic Similarity: Baseline and RAG per PR", fontsize=13, y=1.02)
    fig.tight_layout()
    out = FIGURES_DIR / "fig4_scatter.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


# ── Figure 5: Hallucination score for RAG ────────────────────────────────────

def fig5_hallucination(eval_results):
    all_rag = []
    for _, data in eval_results:
        all_rag.extend([r for r in data["per_pr"]
                         if r["approach"] == "rag" and r.get("hallucination_score", -1) >= 0])

    scores = [r["hallucination_score"] for r in all_rag]

    fig, ax = plt.subplots(figsize=(7, 4))
    bins = [0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]
    ax.hist(scores, bins=bins, color=RAG_COLOR, edgecolor="white", alpha=0.85, rwidth=0.85)

    # Color perfect (1.0) and zero bars differently
    for patch, left_edge in zip(ax.patches, bins[:-1]):
        if left_edge >= 0.9:
            patch.set_facecolor("#2CA02C")   # green = fully grounded
        elif left_edge < 0.1:
            patch.set_facecolor(ACCENT_RED)  # red = fully hallucinated

    ax.set_xlabel("Hallucination Score  (1.0 = all citations grounded, 0.0 = all hallucinated)")
    ax.set_ylabel("Number of PRs")
    ax.set_title(f"Fig 5 — RAG Hallucination Score Distribution\n"
                 f"(n={len(scores)}, mean={np.mean(scores):.3f}, "
                 f"pct fully-grounded={100*sum(s==1.0 for s in scores)/len(scores):.0f}%)")

    green = mpatches.Patch(color="#2CA02C", label="Fully grounded (1.0)")
    red   = mpatches.Patch(color=ACCENT_RED, label="Fully hallucinated (0.0)")
    mid   = mpatches.Patch(color=RAG_COLOR,  label="Partial (0 < x < 1)")
    ax.legend(handles=[green, mid, red])

    fig.tight_layout()
    out = FIGURES_DIR / "fig5_hallucination.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


# ── Figure 6: Generated vs Human Issue Counts ────────────────────────────────

def fig6_issue_counts(eval_results):
    all_rows = []
    for _, data in eval_results:
        all_rows.extend(data["per_pr"])

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, approach, color in [(axes[0], "baseline", BASELINE_COLOR),
                                  (axes[1], "rag",      RAG_COLOR)]:
        pts = [r for r in all_rows if r["approach"] == approach]
        gen    = [r["n_issues_generated"] for r in pts]
        human  = [r["n_human_comments"]   for r in pts]

        ax.scatter(human, gen, alpha=0.45, s=25, color=color,
                   edgecolors="white", linewidths=0.4)
        # y=x reference line
        lim = max(max(human), max(gen)) + 1
        ax.plot([0, lim], [0, lim], "k--", linewidth=1, alpha=0.4, label="Equal line")
        ax.set_xlabel("Human Review Comments")
        ax.set_ylabel("Model Issues Generated")
        ax.set_title(f"{approach.capitalize()}\n"
                     f"avg generated={np.mean(gen):.1f}, avg human={np.mean(human):.1f}")
        ax.legend()

    fig.suptitle("Fig 6 — Generated Issue Count vs Human Comment Count per PR\n"
                 "(points below dashed line = model raised fewer issues than humans)", fontsize=13)
    fig.tight_layout()
    out = FIGURES_DIR / "fig6_issue_counts.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


# ── Figure 7: Qualitative example table ──────────────────────────────────────

def fig7_qualitative_table(eval_results):
    """
    Find 3 interesting PRs (best baseline, worst baseline, best RAG) and render
    a side-by-side summary table as a figure.
    """
    # Load raw baseline and RAG results for django (has both) for text
    baseline_path = Path("data/processed/django/django/baseline_results.json")
    rag_path      = Path("data/processed/django/django/rag_results.json")

    if not baseline_path.exists() or not rag_path.exists():
        print("Skipping fig7 — django results not found")
        return

    with open(baseline_path) as f:
        b_raw = {r["pr_id"]: r for r in json.load(f)}
    with open(rag_path) as f:
        r_raw = {r["pr_id"]: r for r in json.load(f)}

    # Get per-PR scores for django
    django_data = next((d for repo, d in eval_results if "django" in repo), None)
    if not django_data:
        print("Skipping fig7 — django evaluation_results not loaded")
        return

    per_pr = django_data["per_pr"]
    baseline_scores = {r["pr_id"]: r for r in per_pr if r["approach"] == "baseline"}
    rag_scores      = {r["pr_id"]: r for r in per_pr if r["approach"] == "rag"}

    # Pick examples: highest baseline sim, lowest baseline sim, biggest RAG improvement
    shared = sorted(set(baseline_scores) & set(rag_scores))
    by_baseline_sim = sorted(shared, key=lambda i: baseline_scores[i]["semantic_similarity"])
    worst_pr = by_baseline_sim[0]
    best_pr  = by_baseline_sim[-1]
    # biggest RAG gain
    gains = {i: rag_scores[i]["semantic_similarity"] - baseline_scores[i]["semantic_similarity"]
             for i in shared}
    most_improved_pr = max(gains, key=lambda k: gains[k])

    examples = [
        ("Best Baseline PR",      best_pr,          "High sem-sim baseline"),
        ("Worst Baseline PR",     worst_pr,          "Low sem-sim baseline"),
        ("Most RAG-Improved PR",  most_improved_pr,  "Largest RAG gain over baseline"),
    ]

    def _truncate(text, n=120):
        return text[:n] + "…" if len(text) > n else text

    rows = []
    for label, pr_id, note in examples:
        bsc = baseline_scores.get(pr_id, {})
        rsc = rag_scores.get(pr_id, {})
        b_rev = b_raw.get(pr_id, {}).get("review", {})
        r_rev = r_raw.get(pr_id, {}).get("review", {})

        b_summary = _truncate(b_rev.get("summary", "N/A"))
        r_summary = _truncate(r_rev.get("summary", "N/A"))
        b_issues  = b_rev.get("issues", [])
        r_issues  = r_rev.get("issues", [])

        b_top_issue = _truncate(b_issues[0].get("description", "—") if b_issues else "—", 100)
        r_top_issue = _truncate(r_issues[0].get("description", "—") if r_issues else "—", 100)

        rows.append({
            "label":     label,
            "note":      note,
            "pr_id":     pr_id,
            "b_sim":     f"{bsc.get('semantic_similarity', 0):.3f}",
            "r_sim":     f"{rsc.get('semantic_similarity', 0):.3f}",
            "b_rouge":   f"{bsc.get('rouge_l', 0):.3f}",
            "r_rouge":   f"{rsc.get('rouge_l', 0):.3f}",
            "b_halluc":  "N/A",
            "r_halluc":  f"{rsc.get('hallucination_score', 0):.3f}",
            "b_summary": b_summary,
            "r_summary": r_summary,
            "b_issue1":  b_top_issue,
            "r_issue1":  r_top_issue,
        })

    # Render as table figure
    col_labels = ["PR ID", "Metric", "Baseline", "RAG (Δ)"]
    fig, axes = plt.subplots(len(rows), 1, figsize=(14, 4 * len(rows)))

    for ax, row in zip(axes, rows):
        ax.axis("off")
        delta_sim   = float(row["r_sim"])   - float(row["b_sim"])
        delta_rouge = float(row["r_rouge"]) - float(row["b_rouge"])

        table_data = [
            [f"PR #{row['pr_id']}", "Semantic Similarity",
             row["b_sim"],
             f"{row['r_sim']} ({delta_sim:+.3f})"],
            ["", "ROUGE-L",
             row["b_rouge"],
             f"{row['r_rouge']} ({delta_rouge:+.3f})"],
            ["", "Hallucination",
             row["b_halluc"],
             row["r_halluc"]],
            ["Summary", "Baseline →", row["b_summary"], ""],
            ["",        "RAG →",      row["r_summary"], ""],
            ["Top Issue", "Baseline →", row["b_issue1"], ""],
            ["",          "RAG →",      row["r_issue1"], ""],
        ]

        tbl = ax.table(
            cellText=table_data,
            colLabels=col_labels,
            loc="center",
            cellLoc="left",
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(8.5)
        tbl.scale(1, 1.8)

        # Header row color
        for j in range(len(col_labels)):
            tbl[0, j].set_facecolor("#2D6A9F")
            tbl[0, j].set_text_props(color="white", fontweight="bold")
        # Metric rows
        for i in [1, 2, 3]:
            tbl[i, 1].set_facecolor("#EFF3FB")

        ax.set_title(f"{row['label']}  —  {row['note']}",
                     fontsize=11, fontweight="bold", pad=8)

    fig.suptitle("Fig 7 — Qualitative Comparison: Baseline vs RAG (django/django)",
                 fontsize=13, y=1.01)
    fig.tight_layout()
    out = FIGURES_DIR / "fig7_qualitative_table.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


# ── Figure 8: Heatmap of average metrics across repos ────────────────────────

def fig8_heatmap(eval_results, vuln_data):
    """
    One row per repo, columns = metrics.  Combines both evaluation repos
    and the vulnerability accuracy data.
    """
    repo_metrics = {}

    # From evaluation_results
    for repo, data in eval_results:
        summary = data["summary"]
        for approach in ["baseline", "rag"]:
            if approach not in summary:
                continue
            key = f"{repo}\n({approach})"
            m   = summary[approach]
            # Replace -1.0 hallucination with NaN for heatmap
            h = m.get("hallucination_score", None)
            repo_metrics[key] = {
                "Sem. Sim.":    m.get("semantic_similarity", np.nan),
                "ROUGE-L":      m.get("rouge_l", np.nan),
                "Issue Det.":   m.get("issue_detection_score", np.nan),
                "Hallucination": h if (h is not None and h >= 0) else np.nan,
            }

    # From vuln accuracy — add detection rate as a separate repo-level metric
    for r in vuln_data["per_repo"]:
        key = f"{r['repo']}\n(human det.)"
        repo_metrics[key] = {
            "Sem. Sim.":    np.nan,
            "ROUGE-L":      np.nan,
            "Issue Det.":   np.nan,
            "Hallucination": np.nan,
            "Vuln Det. Rate": r["detection_rate"],
        }

    all_cols = ["Sem. Sim.", "ROUGE-L", "Issue Det.", "Hallucination", "Vuln Det. Rate"]
    row_labels = list(repo_metrics.keys())
    matrix = np.full((len(row_labels), len(all_cols)), np.nan)

    for i, key in enumerate(row_labels):
        for j, col in enumerate(all_cols):
            matrix[i, j] = repo_metrics[key].get(col, np.nan)

    fig, ax = plt.subplots(figsize=(11, max(5, len(row_labels) * 0.7 + 2)))

    # Mask NaN for display
    masked = np.ma.masked_invalid(matrix)
    cmap = plt.cm.YlGn.copy()
    cmap.set_bad(color="#F0F0F0")

    im = ax.imshow(masked, aspect="auto", cmap=cmap, vmin=0, vmax=1)

    ax.set_xticks(range(len(all_cols)))
    ax.set_xticklabels(all_cols, rotation=25, ha="right")
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=9)

    # Annotate cells
    for i in range(len(row_labels)):
        for j in range(len(all_cols)):
            val = matrix[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                        fontsize=8, color="black" if val < 0.7 else "white")
            else:
                ax.text(j, i, "—", ha="center", va="center", fontsize=8, color="#AAAAAA")

    plt.colorbar(im, ax=ax, fraction=0.02, pad=0.03, label="Score (0 – 1)")
    ax.set_title("Fig 8 — Metric Heatmap: All Repos & Approaches\n"
                 "(grey cells = metric not applicable)", fontsize=12)
    fig.tight_layout()
    out = FIGURES_DIR / "fig8_per_repo_heatmap.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


# ── Figure 9: RAG vs Baseline per-PR delta histogram ─────────────────────────

def fig9_delta_histogram(eval_results):
    """
    For each PR that has BOTH approaches, show the Δ(RAG − Baseline)
    distribution for semantic similarity and ROUGE-L.
    """
    all_deltas_sim   = []
    all_deltas_rouge = []

    for _, data in eval_results:
        b_map = {r["pr_id"]: r for r in data["per_pr"] if r["approach"] == "baseline"}
        r_map = {r["pr_id"]: r for r in data["per_pr"] if r["approach"] == "rag"}
        for pid in set(b_map) & set(r_map):
            all_deltas_sim.append(r_map[pid]["semantic_similarity"] - b_map[pid]["semantic_similarity"])
            all_deltas_rouge.append(r_map[pid]["rouge_l"] - b_map[pid]["rouge_l"])

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    for ax, deltas, metric in [(axes[0], all_deltas_sim,   "Semantic Similarity"),
                                 (axes[1], all_deltas_rouge, "ROUGE-L")]:
        pos = sum(d > 0 for d in deltas)
        neg = sum(d < 0 for d in deltas)
        ax.hist(deltas, bins=20, color=RAG_COLOR, edgecolor="white", alpha=0.8)
        ax.axvline(0, color="black", linewidth=1.4, linestyle="--", label="No change")
        ax.axvline(np.mean(deltas), color=BASELINE_COLOR, linewidth=1.4,
                   linestyle="-", label=f"Mean Δ = {np.mean(deltas):+.4f}")
        ax.set_xlabel(f"Δ {metric}  (RAG − Baseline)")
        ax.set_ylabel("PR Count")
        ax.set_title(f"RAG − Baseline Δ {metric}\n"
                     f"RAG better: {pos} PRs | Baseline better: {neg} PRs")
        ax.legend()

    fig.suptitle("Fig 9 — Per-PR Improvement: RAG vs Baseline\n"
                 "(positive = RAG outperforms baseline on that PR)", fontsize=13)
    fig.tight_layout()
    out = FIGURES_DIR / "fig9_delta_histogram.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    os.makedirs("figures", exist_ok=True)
    print("Loading data …")
    vuln_data    = load_vuln_accuracy()
    eval_results = load_eval_results()

    if not eval_results:
        print("WARNING: No evaluation_results.json files found under data/processed/")

    print(f"Loaded: vuln_accuracy ({len(vuln_data['per_repo'])} repos), "
          f"eval_results ({len(eval_results)} repos)")

    print("\nGenerating figures …")
    fig1_vuln_detection_rate(vuln_data)

    if eval_results:
        fig2_metric_comparison(eval_results)
        fig3_distributions(eval_results)
        fig4_scatter(eval_results)
        fig5_hallucination(eval_results)
        fig6_issue_counts(eval_results)
        fig7_qualitative_table(eval_results)
        fig8_heatmap(eval_results, vuln_data)
        fig9_delta_histogram(eval_results)

    print(f"\nAll figures saved to {FIGURES_DIR.resolve()}/")


if __name__ == "__main__":
    main()
