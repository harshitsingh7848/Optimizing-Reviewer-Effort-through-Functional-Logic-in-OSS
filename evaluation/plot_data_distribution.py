"""
analyse_thresholds.py
---------------------
Helps you decide where to draw the "high effort" line for ground truth.

For BOTH metrics (total_comments and review_duration_hours) it shows:
  - The full distribution (histogram)
  - A threshold sensitivity table: at Q70/Q75/Q80/Q85/Q90, how many PRs
    are labelled high-effort and what the cutoff value is
  - A cumulative % curve so you can visually pick a threshold

Usage:
    python analyse_thresholds.py pr_dataset.json
    python analyse_thresholds.py pr_dataset_pandas.json pr_dataset_django.json
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

OUT_DIR = Path("figures")
PERCENTILES_TO_CHECK = [70, 75, 80, 85, 90]
COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]


def load(path: str) -> list[dict]:
    data = json.loads(Path(path).read_text("utf-8"))
    print(f"  Loaded {len(data)} PRs  ← {path}")
    return data



def print_threshold_table(vals: list[float], metric: str, label: str) -> None:
    arr = np.array(vals)
    print(f"\n  ── {label}  /  {metric} ───────────────────────────────")
    print(f"  {'Percentile':>12}  {'Cutoff value':>14}  "
          f"{'High-effort n':>14}  {'High-effort %':>14}")
    print(f"  {'-'*58}")
    for p in PERCENTILES_TO_CHECK:
        cutoff   = np.percentile(arr, p)
        n_high   = int((arr >= cutoff).sum())
        pct_high = n_high / len(arr) * 100
        marker   = "  ← suggested" if p == 80 else ""
        print(f"  {f'Q{p}':>12}  {cutoff:>14.2f}  {n_high:>14d}  "
              f"{pct_high:>13.1f}%{marker}")
    print(f"\n  Total PRs : {len(arr)}")
    print(f"  Mean      : {arr.mean():.2f}")
    print(f"  Median    : {np.median(arr):.2f}")
    print(f"  Zero-value: {(arr == 0).sum()} PRs "
          f"({(arr == 0).mean()*100:.1f}%  — these will always be low-effort)")



def plot_dataset(data: list[dict], label: str, color: str, out_path: Path) -> None:

    comments  = np.array([r.get("total_comments", 0)         for r in data])
    durations = np.array([r.get("review_duration_hours", 0)  for r in data])

    fig = plt.figure(figsize=(14, 9))
    fig.suptitle(f"Threshold Analysis — {label}  (n={len(data)})",
                 fontsize=14, fontweight="bold")
    gs = gridspec.GridSpec(2, 2, hspace=0.45, wspace=0.35)

    for row_idx, (arr, metric_label, unit) in enumerate([
        (comments,  "Comment Count",          "comments"),
        (durations, "Review Duration",         "hours"),
    ]):
        ax_hist = fig.add_subplot(gs[row_idx, 0])
        ax_cdf  = fig.add_subplot(gs[row_idx, 1])

        # ── Histogram ──────────────────────────────────────────────────────
        cap     = np.percentile(arr, 99) if len(arr) > 1 else arr.max()
        clipped = np.clip(arr, 0, cap)
        n_bins  = min(50, max(10, len(arr) // 8))

        ax_hist.hist(clipped, bins=n_bins, color=color, alpha=0.7, edgecolor="white")

        line_styles = [
            (75, "--", "red",    f"Q75 = {np.percentile(arr,75):.1f}"),
            (80, ":",  "purple", f"Q80 = {np.percentile(arr,80):.1f}"),
        ]
        for pct, ls, lc, lbl in line_styles:
            ax_hist.axvline(np.percentile(arr, pct), color=lc,
                            linestyle=ls, linewidth=1.8, label=lbl)

        ax_hist.set_title(f"{metric_label} — Histogram", fontweight="bold")
        ax_hist.set_xlabel(f"{metric_label} ({unit})")
        ax_hist.set_ylabel("Number of PRs")
        ax_hist.legend(fontsize=8)

        # Shade the "high effort" region (above Q80) in the histogram
        q80 = np.percentile(arr, 80)
        ax_hist.axvspan(q80, cap, alpha=0.12, color="purple",
                        label="_high effort zone")

        # ── Cumulative % curve (shows what % is below each threshold) ──────
        sorted_arr = np.sort(arr)
        cdf        = np.arange(1, len(sorted_arr) + 1) / len(sorted_arr) * 100

        ax_cdf.plot(sorted_arr, cdf, color=color, linewidth=2)

        for pct in PERCENTILES_TO_CHECK:
            cutoff   = np.percentile(arr, pct)
            pct_high = 100 - pct          # % labelled HIGH effort
            ax_cdf.scatter(cutoff, pct, color="grey", s=40, zorder=5)
            ax_cdf.annotate(
                f"Q{pct}: {cutoff:.0f} {unit}\n→ {pct_high:.0f}% high",
                xy=(cutoff, pct),
                xytext=(8, -12),
                textcoords="offset points",
                fontsize=7.5,
                color="dimgrey",
            )

        # Highlight Q75 and Q80
        for pct, lc, ls in [(75, "red", "--"), (80, "purple", ":")]:
            cutoff = np.percentile(arr, pct)
            ax_cdf.axhline(pct, color=lc, linestyle=ls, linewidth=1.2, alpha=0.7)
            ax_cdf.axvline(cutoff, color=lc, linestyle=ls, linewidth=1.2, alpha=0.7)

        ax_cdf.set_title(f"{metric_label} — Cumulative % (below threshold)",
                         fontweight="bold")
        ax_cdf.set_xlabel(f"{metric_label} ({unit})")
        ax_cdf.set_ylabel("Cumulative % of PRs")
        ax_cdf.set_ylim(0, 105)
        ax_cdf.grid(True, alpha=0.3)

        # ── Console table ───────────────────────────────────────────────────
        print_threshold_table(arr.tolist(), metric_label, label)

    OUT_DIR.mkdir(exist_ok=True)
    slug = label.replace("/", "_").replace(" ", "_")
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"\n  Saved → {out_path}")



def plot_comparison(datasets: dict[str, list[dict]]) -> None:
    """One figure with all datasets overlaid — easy to compare repos."""
    labels = list(datasets.keys())
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Comments vs Duration Threshold Comparison — All Datasets",
                 fontsize=13, fontweight="bold")

    for ax, (metric_key, metric_label, unit) in zip(axes, [
        ("total_comments",        "Comment Count",   "comments"),
        ("review_duration_hours", "Review Duration", "hours"),
    ]):
        for (label, data), color in zip(datasets.items(), COLORS):
            arr    = np.array([r.get(metric_key, 0) for r in data])
            sorted_arr = np.sort(arr)
            cdf        = np.arange(1, len(sorted_arr) + 1) / len(sorted_arr) * 100
            ax.plot(sorted_arr, cdf, color=color, linewidth=2, label=label)

            # Mark Q80 for each dataset
            q80 = np.percentile(arr, 80)
            ax.scatter(q80, 80, color=color, s=60, zorder=6)
            ax.annotate(f"Q80={q80:.0f}", xy=(q80, 80),
                        xytext=(5, 4), textcoords="offset points",
                        fontsize=8, color=color)

        ax.axhline(80, color="purple", linestyle=":", linewidth=1.2,
                   alpha=0.6, label="Q80 (top 20%)")
        ax.axhline(75, color="red",    linestyle="--", linewidth=1.2,
                   alpha=0.6, label="Q75 (top 25%)")
        ax.set_xlabel(f"{metric_label} ({unit})")
        ax.set_ylabel("Cumulative % of PRs")
        ax.set_title(metric_label, fontweight="bold")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 105)

    OUT_DIR.mkdir(exist_ok=True)
    out = OUT_DIR / "fig_threshold_comparison.png"
    fig.savefig(out, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"\n  Comparison saved → {out}")


if __name__ == "__main__":
    paths = sys.argv[1:] or ["pr_dataset_pandas.json"]
    

    print("Loading datasets...")
    datasets: dict[str, list[dict]] = {}
    for p in paths:
        label = Path(p).stem.replace("pr_dataset_", "").replace("_", "/", 1)
        datasets[label] = load(p)

    for (label, data), color in zip(datasets.items(), COLORS):
        out = OUT_DIR / f"pandas_fig_thresholds_{label.replace('/', '_')}.png"
        print(f"\n{'='*60}")
        print(f"  Analysing: {label}")
        print(f"{'='*60}")
        plot_dataset(data, label, color, out)

    if len(datasets) >= 2:
        print(f"\n{'='*60}")
        print(f"  Cross-dataset comparison")
        print(f"{'='*60}")
        plot_comparison(datasets)

    print("\nDone.")