import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from collections import Counter

NAVY  = "#065A82"
TEAL  = "#1C7293"
ACCENT = "#F59E0B"
RED   = "#DC2626"
GREEN = "#16A34A"
GRAY  = "#64748B"
LGRAY = "#E2EAF0"

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


def load_results(path: str = "llm_judge_results.json") -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def get_display_name(model_key: str) -> str:
    return DISPLAY_NAMES.get(model_key, model_key)


def plot_metrics(data: dict, output: str = "metrics.png"):
    metrics_data = data["metrics"]
    all_results = data.get("results", [])

    model_keys = [k for k in metrics_data if "llm_j" in metrics_data[k]]
    if not model_keys:
        print("No model metrics found, skipping plot 1.")
        return

    labels = [get_display_name(k) for k in model_keys] + ["RF Baseline"]

    first_model = metrics_data[model_keys[0]]
    rf = first_model.get("random_forest_baseline", {})

    rows = []
    for k in model_keys:
        m = metrics_data[k]["llm_j"]
        rows.append({
            "precision": m["precision"],
            "recall":    m["recall"],
            "f1":        m["f1"],
        })
    rows.append({
        "precision": rf.get("precision", 0),
        "recall":    rf.get("recall", 0),
        "f1":        rf.get("f1", 0),
    })

    n_total = first_model.get("n_prs_evaluated", "?")

    fig, ax = plt.subplots(figsize=(10, 5.5))
    metric_names  = ["precision", "recall", "f1"]
    metric_labels = ["Precision", "Recall", "F1-Score"]
    colors = [NAVY, TEAL, ACCENT]
    x = np.arange(len(labels))
    width = 0.22

    for i, (metric, label, color) in enumerate(zip(metric_names, metric_labels, colors)):
        vals = [r[metric] for r in rows]
        bars = ax.bar(x + i * width, vals, width, label=label,
                      color=color, zorder=3, edgecolor="white", linewidth=0.5)
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
        f"RQ1: Precision, Recall and F1-Score by Model\n"
        f"(High-Effort Class, pandas-dev/pandas, n={n_total} PRs)",
        fontsize=12, fontweight="bold", pad=12,
    )
    ax.yaxis.grid(True, color=LGRAY, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="upper right", fontsize=10, framealpha=0.9)
    plt.tight_layout()
    plt.savefig(output, dpi=150, bbox_inches="tight")
    plt.close()


def plot_circularity(data: dict, output: str = "circularity_check.png"):
    metrics_data = data["metrics"]


    first_key = next((k for k in metrics_data if "circularity_check" in metrics_data[k]), None)
    if first_key is None:
        print("No circularity data found, skipping plot 2.")
        return

    circ = metrics_data[first_key]["circularity_check"]
    feat_keys = list(circ.keys())
    rhos = [circ[f]["spearman_rho"] for f in feat_keys]
    sigs = [circ[f]["significant"] for f in feat_keys]
    bar_colors = [RED if s else GREEN for s in sigs]
    feat_labels = [CIRCULARITY_LABELS.get(f, f) for f in feat_keys]

    fig, ax = plt.subplots(figsize=(9, 4.5))
    y = np.arange(len(feat_keys))
    bars = ax.barh(y, rhos, color=bar_colors, height=0.5, zorder=3, edgecolor="white")

    for bar, val in zip(bars, rhos):
        ax.text(val + 0.006, bar.get_y() + bar.get_height() / 2,
                f"ρ = {val:.3f}", va="center", fontsize=9,
                fontweight="bold", color="#1e293b")

    ax.set_yticks(y)
    ax.set_yticklabels(feat_labels, fontsize=11)
    ax.set_xlim(0, 0.45)
    ax.set_xlabel("Spearman ρ with Effort Score", fontsize=11)
    ax.set_title(
        "Circularity Check: Feature Correlation with Ground Truth Effort\n"
        "(Red = significant circularity risk p < 0.05,  Green = no circularity p ≥ 0.05)",
        fontsize=11, fontweight="bold", pad=10,
    )
    ax.xaxis.grid(True, color=LGRAY, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.axvline(0, color=GRAY, linewidth=0.8)

    red_patch   = mpatches.Patch(color=RED,   label="Significant circularity (p < 0.05)")
    green_patch = mpatches.Patch(color=GREEN, label="No circularity (p ≥ 0.05)")
    ax.legend(handles=[red_patch, green_patch], fontsize=9, loc="lower right")
    plt.tight_layout()
    plt.savefig(output, dpi=150, bbox_inches="tight")
    plt.close()


def plot_mcnemar(data: dict, output: str = "mcnemar_results.png"):
    metrics_data = data["metrics"]
    model_keys = [k for k in metrics_data if "mcnemar_table" in metrics_data[k]]
    if not model_keys:
        print("No McNemar data found, skipping plot 3.")
        return

    fig, axes = plt.subplots(1, len(model_keys), figsize=(11, 3.8))
    if len(model_keys) == 1:
        axes = [axes]

    for ax, key in zip(axes, model_keys):
        m = metrics_data[key]
        tbl = m["mcnemar_table"]
        p_val = m["mcnemar_p_value"]
        sig = p_val < 0.05

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

        sig_str = "Significant" if sig else "Not significant"
        sig_color = RED if sig else GREEN
        ax.set_title(
            f"{get_display_name(key)}\np = {p_val} {sig_str}",
            fontsize=9, fontweight="bold", color=sig_color,
        )

    plt.suptitle("McNemar Statistic: LLM-J vs RF Baseline",
                 fontsize=12, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(output, dpi=150, bbox_inches="tight")
    plt.close()


def plot_risk_distribution(data: dict, output: str = "risk_distribution.png"):
    metrics_data = data["metrics"]
    all_results = data.get("results", [])
    model_keys = [k for k in metrics_data if "llm_j" in metrics_data[k]]
    if not model_keys or not all_results:
        print("No results data found, skipping plot 4.")
        return

    fig, axes = plt.subplots(1, len(model_keys), figsize=(11, 4))
    if len(model_keys) == 1:
        axes = [axes]

    for ax, key in zip(axes, model_keys):
        model_results = [r for r in all_results
                         if r.get("model_name") == key
                         and isinstance(r.get("risk_score"), int)]
        score_counts = Counter(r["risk_score"] for r in model_results)

        scores = sorted(score_counts.keys())
        counts = [score_counts[s] for s in scores]
        bar_colors = [GREEN if s <= 3 else RED for s in scores]

        bars = ax.bar(scores, counts, color=bar_colors,
                      zorder=3, edgecolor="white", linewidth=0.5)
        for bar, cnt in zip(bars, counts):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 3,
                    str(cnt), ha="center", va="bottom",
                    fontsize=9, fontweight="bold")

        ax.set_xlabel("Risk Score", fontsize=10)
        ax.set_ylabel("Number of PRs", fontsize=10)
        avg = metrics_data[key]["llm_j"]["avg_risk_score"]
        ax.set_title(f"{get_display_name(key)}\n(avg = {avg:.2f}/5)",
                     fontsize=10, fontweight="bold")
        ax.set_xticks(range(1, 6))
        ax.yaxis.grid(True, color=LGRAY, linewidth=0.8, zorder=0)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        ax.axvline(3.5, color=RED, linewidth=1.5, linestyle="--", alpha=0.7,
                   label="Threshold (≥4 = High)")
        if key == model_keys[0]:
            ax.legend(fontsize=7, loc="upper right")

    green_p = mpatches.Patch(color=GREEN, label="Low Effort (1-3)")
    red_p   = mpatches.Patch(color=RED,   label="High Effort (4-5)")
    fig.legend(handles=[green_p, red_p], loc="lower center",
               ncol=2, fontsize=10, bbox_to_anchor=(0.5, -0.05))
    plt.suptitle("Risk Score Distribution by Model\n(Red dashed line = prediction threshold ≥ 4)",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output, dpi=150, bbox_inches="tight")
    plt.close()


def main():
    data = load_results("llm_judge_results.json")
    plot_metrics(data)
    plot_circularity(data)
    plot_mcnemar(data)
    plot_risk_distribution(data)
    print("\nAll plots generated successfully.")


if __name__ == "__main__":
    main()
