"""
plot_results.py
===============
Unified plotting script for RQ1 and RQ2 results.

Usage:
    python plot_results.py --rq1 results/llm_judge_results_pandas.json --repo pandas-dev/pandas
    python plot_results.py --rq1 results/llm_judge_results_django.json --repo django/django
    python plot_results.py --rq2 results/llm_judge_rq2_results.json    --repo django/django
    python plot_results.py --rq1 results/llm_judge_results_django.json \
                           --rq2 results/llm_judge_rq2_results.json    \
                           --repo django/django --outdir figures
"""

import argparse
import json
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import Counter

# ── Palette ───────────────────────────────────────────────────────────────────
NAVY   = "#065A82"
TEAL   = "#1C7293"
ACCENT = "#F59E0B"
RED    = "#DC2626"
GREEN  = "#16A34A"
GRAY   = "#64748B"
LGRAY  = "#E2EAF0"

DISPLAY_NAMES = {
    "qwen2.5:7b-instruct": "Qwen2.5-7B",
    "llama3.1:8b":         "Llama3.1-8B",
    "gemma3:4b":           "Gemma3-4B",
}

CIRCULARITY_LABELS = {
    "additions":        "Additions",
    "deletions":        "Deletions",
    "cyclomatic_delta": "Cyclomatic Δ",
    "max_nesting":      "Max Nesting",
    "logic_density":    "Logic Density",
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def load(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def dn(key: str) -> str:
    return DISPLAY_NAMES.get(key, key)


def model_keys_with(data: dict, field: str) -> list:
    return [k for k, v in data["metrics"].items() if field in v]


def save(fig, path: str):
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ── Plot 1: Precision / Recall / F1 ──────────────────────────────────────────

def plot_metrics(data: dict, rq: int, repo: str, out: str):
    """Works for both RQ1 (PR-level) and RQ2 (segment-level)."""
    md   = data["metrics"]
    keys = model_keys_with(data, "llm_j")
    if not keys:
        print("  [skip] plot_metrics — no llm_j data found.")
        return

    first = md[keys[0]]
    if rq == 1:
        n_total = first.get("n_prs_evaluated", "?")
        unit    = "PRs"
        gt_desc = "High-Effort Class"
    else:
        n_total = first.get("n_segments_evaluated",
                  first.get("n_prs_evaluated", "?"))
        unit    = "segments"
        gt_desc = "High-Risk Segments (inline comment GT)"

    rf = first.get("random_forest_baseline", {})

    rows = []
    for k in keys:
        m = md[k]["llm_j"]
        rows.append({"precision": m["precision"],
                     "recall":    m["recall"],
                     "f1":        m["f1"]})
    rows.append({"precision": rf.get("precision", 0),
                 "recall":    rf.get("recall", 0),
                 "f1":        rf.get("f1", 0)})

    labels        = [dn(k) for k in keys] + ["RF Baseline"]
    metric_names  = ["precision", "recall", "f1"]
    metric_labels = ["Precision", "Recall", "F1-Score"]
    colors        = [NAVY, TEAL, ACCENT]
    x             = np.arange(len(labels))
    width         = 0.22

    fig, ax = plt.subplots(figsize=(10, 5.5))
    for i, (mn, ml, c) in enumerate(zip(metric_names, metric_labels, colors)):
        vals = [r[mn] for r in rows]
        bars = ax.bar(x + i * width, vals, width, label=ml,
                      color=c, zorder=3, edgecolor="white", linewidth=0.5)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.012,
                    f"{val:.2f}", ha="center", va="bottom",
                    fontsize=8, fontweight="bold", color="#1e293b")

    ax.set_xticks(x + width)
    ax.set_xticklabels(labels, fontsize=11, fontweight="bold")
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_title(
        f"RQ{rq}: Precision, Recall and F1-Score by Model\n"
        f"({gt_desc}, {repo}, n={n_total} {unit})",
        fontsize=12, fontweight="bold", pad=12,
    )
    ax.yaxis.grid(True, color=LGRAY, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="upper right", fontsize=10, framealpha=0.9)
    plt.tight_layout()
    save(fig, out)


# ── Plot 2: Circularity check (RQ1 only) ─────────────────────────────────────

def plot_circularity(data: dict, repo: str, out: str):
    md        = data["metrics"]
    first_key = next((k for k in md if "circularity_check" in md[k]), None)
    if first_key is None:
        print("  [skip] plot_circularity — no circularity_check data (RQ2 doesn't have this).")
        return

    circ        = md[first_key]["circularity_check"]
    feat_keys   = list(circ.keys())
    rhos        = [circ[f]["spearman_rho"] for f in feat_keys]
    sigs        = [circ[f]["significant"]  for f in feat_keys]
    bar_colors  = [RED if s else GREEN for s in sigs]
    feat_labels = [CIRCULARITY_LABELS.get(f, f) for f in feat_keys]

    fig, ax = plt.subplots(figsize=(9, 4.5))
    y    = np.arange(len(feat_keys))
    bars = ax.barh(y, rhos, color=bar_colors, height=0.5,
                   zorder=3, edgecolor="white")

    for bar, val in zip(bars, rhos):
        ax.text(val + 0.006, bar.get_y() + bar.get_height() / 2,
                f"ρ = {val:.3f}", va="center", fontsize=9,
                fontweight="bold", color="#1e293b")

    ax.set_yticks(y)
    ax.set_yticklabels(feat_labels, fontsize=11)
    ax.set_xlim(0, max(rhos) * 1.4 + 0.05)
    ax.set_xlabel("Spearman ρ with Effort Score", fontsize=11)
    ax.set_title(
        f"RQ1: Circularity Check — Feature Correlation with Ground Truth\n"
        f"({repo}  |  Red = significant p < 0.05,  Green = no circularity p ≥ 0.05)",
        fontsize=11, fontweight="bold", pad=10,
    )
    ax.xaxis.grid(True, color=LGRAY, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.axvline(0, color=GRAY, linewidth=0.8)
    ax.legend(handles=[
        mpatches.Patch(color=RED,   label="Significant circularity (p < 0.05)"),
        mpatches.Patch(color=GREEN, label="No circularity (p ≥ 0.05)"),
    ], fontsize=9, loc="lower right")
    plt.tight_layout()
    save(fig, out)


# ── Plot 3: McNemar heatmap ───────────────────────────────────────────────────

def plot_mcnemar(data: dict, rq: int, repo: str, out: str):
    md   = data["metrics"]
    keys = model_keys_with(data, "mcnemar_table")
    if not keys:
        print("  [skip] plot_mcnemar — no mcnemar_table data found.")
        return

    fig, axes = plt.subplots(1, len(keys), figsize=(4 * len(keys), 3.8))
    if len(keys) == 1:
        axes = [axes]

    for ax, key in zip(axes, keys):
        m     = md[key]
        tbl   = m["mcnemar_table"]
        p_val = m["mcnemar_p_value"]
        sig   = p_val < 0.05

        grid = np.array([[tbl["a"], tbl["b"]],
                         [tbl["c"], tbl["d"]]])
        vmax = max(grid.flatten()) + 5
        ax.imshow(grid, cmap="Blues", aspect="auto", vmin=0, vmax=vmax)

        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(grid[i, j]),
                        ha="center", va="center", fontsize=14,
                        fontweight="bold",
                        color="white" if grid[i, j] > vmax * 0.5 else "#1e293b")

        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["RF Correct", "RF Wrong"], fontsize=8)
        ax.set_yticklabels(["LLM Correct", "LLM Wrong"], fontsize=8)
        ax.set_title(
            f"{dn(key)}\np = {p_val} ({'Significant' if sig else 'Not significant'})",
            fontsize=9, fontweight="bold",
            color=RED if sig else GREEN,
        )

    plt.suptitle(
        f"RQ{rq}: McNemar Test — LLM-J vs RF Baseline  ({repo})",
        fontsize=12, fontweight="bold", y=1.02,
    )
    plt.tight_layout()
    save(fig, out)


# ── Plot 4: Risk score distribution ──────────────────────────────────────────

def plot_risk_distribution(data: dict, rq: int, repo: str, out: str):
    md          = data["metrics"]
    all_results = data.get("results", [])
    keys        = model_keys_with(data, "llm_j")
    if not keys or not all_results:
        print("  [skip] plot_risk_distribution — no results data found.")
        return

    unit   = "PRs" if rq == 1 else "Segments"
    ylabel = f"Number of {unit}"

    fig, axes = plt.subplots(1, len(keys), figsize=(4 * len(keys), 4))
    if len(keys) == 1:
        axes = [axes]

    for ax, key in zip(axes, keys):
        model_results = [r for r in all_results
                         if r.get("model_name") == key
                         and isinstance(r.get("risk_score"), int)]
        score_counts = Counter(r["risk_score"] for r in model_results)
        scores  = sorted(score_counts)
        counts  = [score_counts[s] for s in scores]
        colors  = [GREEN if s <= 3 else RED for s in scores]

        bars = ax.bar(scores, counts, color=colors,
                      zorder=3, edgecolor="white", linewidth=0.5)
        for bar, cnt in zip(bars, counts):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + max(counts) * 0.02,
                    str(cnt), ha="center", va="bottom",
                    fontsize=9, fontweight="bold")

        ax.set_xlabel("Risk Score", fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        avg = md[key]["llm_j"]["avg_risk_score"]
        ax.set_title(f"{dn(key)}\n(avg = {avg:.2f}/5)",
                     fontsize=10, fontweight="bold")
        ax.set_xticks(range(1, 6))
        ax.yaxis.grid(True, color=LGRAY, linewidth=0.8, zorder=0)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        ax.axvline(3.5, color=RED, linewidth=1.5, linestyle="--",
                   alpha=0.7, label="Threshold (≥4 = High)")
        if key == keys[0]:
            ax.legend(fontsize=7, loc="upper right")

    fig.legend(handles=[
        mpatches.Patch(color=GREEN, label="Low Risk (1-3)"),
        mpatches.Patch(color=RED,   label="High Risk (4-5)"),
    ], loc="lower center", ncol=2, fontsize=10, bbox_to_anchor=(0.5, -0.05))

    plt.suptitle(
        f"RQ{rq}: Risk Score Distribution by Model  ({repo})\n"
        f"(Red dashed line = prediction threshold ≥ 4)",
        fontsize=12, fontweight="bold",
    )
    plt.tight_layout()
    save(fig, out)


# ── Plot 5: RQ2-only — high-risk vs low-risk segment counts ──────────────────

def plot_rq2_segment_breakdown(data: dict, repo: str, out: str):
    md          = data["metrics"]
    all_results = data.get("results", [])
    keys        = model_keys_with(data, "llm_j")
    if not keys:
        return

    labels    = [dn(k) for k in keys]
    gt_high   = [md[k]["n_high_risk"] for k in keys]   # same for all — 207
    pred_high = []
    for k in keys:
        model_results = [r for r in all_results
                         if r.get("model_name") == k
                         and isinstance(r.get("risk_score"), int)]
        pred_high.append(sum(1 for r in model_results if r["risk_score"] >= 4))

    x     = np.arange(len(keys))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 4.5))
    b1 = ax.bar(x - width/2, gt_high,   width, label="Actual High-Risk (GT)",
                color=NAVY, zorder=3, edgecolor="white")
    b2 = ax.bar(x + width/2, pred_high, width, label="Predicted High-Risk (LLM)",
                color=RED,  zorder=3, edgecolor="white")

    for bars in (b1, b2):
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 2,
                    str(int(bar.get_height())),
                    ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11, fontweight="bold")
    ax.set_ylabel("Number of Segments", fontsize=11)
    ax.set_title(
        f"RQ2: Actual vs Predicted High-Risk Segments  ({repo})\n"
        f"(Threshold ≥ 4 for LLM prediction  |  GT = inline reviewer comment)",
        fontsize=11, fontweight="bold", pad=10,
    )
    ax.yaxis.grid(True, color=LGRAY, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=10)
    plt.tight_layout()
    save(fig, out)


# ── Runners ───────────────────────────────────────────────────────────────────

def run_rq1(path: str, repo: str, prefix: str):
    print(f"\n── RQ1 plots from {path} ──")
    data = load(path)
    plot_metrics(data,           rq=1, repo=repo, out=f"{prefix}_metrics.png")
    plot_circularity(data,              repo=repo, out=f"{prefix}_circularity.png")
    plot_mcnemar(data,           rq=1, repo=repo, out=f"{prefix}_mcnemar.png")
    plot_risk_distribution(data, rq=1, repo=repo, out=f"{prefix}_risk_dist.png")


def run_rq2(path: str, repo: str, prefix: str):
    print(f"\n── RQ2 plots from {path} ──")
    data = load(path)
    plot_metrics(data,                  rq=2, repo=repo, out=f"{prefix}_metrics.png")
    plot_mcnemar(data,                  rq=2, repo=repo, out=f"{prefix}_mcnemar.png")
    plot_risk_distribution(data,        rq=2, repo=repo, out=f"{prefix}_risk_dist.png")
    plot_rq2_segment_breakdown(data,          repo=repo, out=f"{prefix}_segment_breakdown.png")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Plot RQ1 and/or RQ2 results.")
    parser.add_argument("--rq1",    default=None,
                        help="Path to RQ1 results JSON "
                             "(e.g. results/llm_judge_results_django.json)")
    parser.add_argument("--rq2",    default=None,
                        help="Path to RQ2 results JSON "
                             "(e.g. results/llm_judge_rq2_results.json)")
    parser.add_argument("--repo",   default="django/django",
                        help="Repo label for plot titles (e.g. django/django)")
    parser.add_argument("--outdir", default="figures",
                        help="Root output directory — rq1/ and rq2/ subdirs "
                             "are created automatically (default: figures/)")
    args = parser.parse_args()

    if not args.rq1 and not args.rq2:
        parser.error("Provide at least one of --rq1 or --rq2.")

    repo      = args.repo
    repo_slug = repo.replace("/", "_").replace("-", "_")

    if args.rq1:
        out = os.path.join(args.outdir, "rq1")
        os.makedirs(out, exist_ok=True)
        run_rq1(args.rq1, repo, prefix=os.path.join(out, f"rq1_{repo_slug}"))

    if args.rq2:
        out = os.path.join(args.outdir, "rq2")
        os.makedirs(out, exist_ok=True)
        run_rq2(args.rq2, repo, prefix=os.path.join(out, f"rq2_{repo_slug}"))

    print("\nDone.")


if __name__ == "__main__":
    main()