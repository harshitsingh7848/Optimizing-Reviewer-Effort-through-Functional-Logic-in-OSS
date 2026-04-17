import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

NAVY  = "#21295C"
DARK  = "#065A82"
TEAL  = "#1C7293"
RED   = "#E63946"
GOLD  = "#F59E0B"
LGRAY = "#E5E5E5"
WHITE = "#FFFFFF"

MODELS = ["Qwen2.5-7B", "Llama3.1-8B", "Gemma3-4B"]
RHO    = [-0.0835,      -0.0113,        -0.0475]
PVALS  = [ 0.0622,       0.8012,         0.2890]

def save(fig, path):
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {path}")

fig, ax = plt.subplots(figsize=(8, 4.5))
labels   = ["Revert-Only\n(Precise)", "Bugfix/Security\nFile-Overlap (Broad)"]
counts   = [0, 140]
totals   = [500, 500]
pct      = [0.0, 28.0]
colors   = [TEAL, RED]

bars = ax.bar(labels, counts, color=colors, width=0.45, zorder=3, edgecolor="white")
for bar, c, p in zip(bars, counts, pct):
    ax.text(bar.get_x() + bar.get_width()/2,
            bar.get_height() + 4,
            f"{c} PRs ({p:.0f}%)",
            ha="center", va="bottom", fontsize=11, fontweight="bold")

ax.set_ylabel("Positive Outcomes Found", fontsize=11)
ax.set_title("RQ3: Ground Truth Comparison - django/django (n=500 PRs)\n"
             "Two labeling strategies, both yielding inconclusive results",
             fontsize=11, fontweight="bold", pad=10)
ax.set_ylim(0, 200)
ax.yaxis.grid(True, color=LGRAY, linewidth=0.8, zorder=0)
ax.set_axisbelow(True)
ax.spines[["top", "right"]].set_visible(False)

ax.annotate("Django uses\nforward-fix culture\n→ 0 explicit reverts",
            xy=(0, 2), xytext=(0.15, 80),
            fontsize=9, color=TEAL, ha="center",
            arrowprops=dict(arrowstyle="->", color=TEAL, lw=1.2))
ax.annotate("Shared core files\ncause false attribution\n→ GT unreliable",
            xy=(1, 142), xytext=(0.85, 175),
            fontsize=9, color=RED, ha="center",
            arrowprops=dict(arrowstyle="->", color=RED, lw=1.2))

plt.tight_layout()
save(fig, "../figures/rq3/rq3_gt_comparison.png")


fig, ax = plt.subplots(figsize=(8, 4.5))
x = np.arange(len(MODELS))
bar_colors = [GOLD if p < 0.05 else TEAL for p in PVALS]
bars = ax.bar(x, RHO, color=bar_colors, width=0.5, zorder=3, edgecolor="white")

for bar, rho, p in zip(bars, RHO, PVALS):
    label = f"ρ={rho:.3f}\np={p:.3f}"
    y_pos = rho - 0.005 if rho < 0 else rho + 0.002
    ax.text(bar.get_x() + bar.get_width()/2, y_pos,
            label, ha="center", va="top" if rho < 0 else "bottom",
            fontsize=10, fontweight="bold", color="white" if abs(rho) > 0.04 else NAVY)

ax.axhline(0, color="black", linewidth=0.8)
ax.axhline(0.1,  color=RED, linewidth=1, linestyle="--", alpha=0.5, label="Weak positive (ρ=0.10)")
ax.axhline(-0.1, color=RED, linewidth=1, linestyle="--", alpha=0.5)
ax.set_xticks(x)
ax.set_xticklabels(MODELS, fontsize=11, fontweight="bold")
ax.set_ylabel("Spearman ρ  (LLM Risk Score vs Post-Merge Outcome)", fontsize=10)
ax.set_title("RQ3: Spearman Correlation — LLM Risk Score vs Post-Merge Outcome\n"
             "(File-overlap GT, n=500, 140 positive outcomes — all p > 0.05)",
             fontsize=11, fontweight="bold", pad=10)
ax.set_ylim(-0.15, 0.15)
ax.yaxis.grid(True, color=LGRAY, linewidth=0.8, zorder=0)
ax.set_axisbelow(True)
ax.spines[["top","right"]].set_visible(False)
ax.legend(fontsize=9)
plt.tight_layout()
save(fig, "../figures/rq3/rq3_spearman.png")



with open("../data/rq3_outcome_labels.json") as f:
    labels_data = json.load(f)
with open("../results/llm_judge_results_django.json") as f:
    rq1 = json.load(f)

outcome_map = {r["pr_id"]: r["outcome"] for r in labels_data}

fig, axes = plt.subplots(1, 3, figsize=(13, 4.5), sharey=False)
fig.suptitle("RQ3: Risk Score Distribution by Outcome  (django/django)\n"
             "(File-overlap GT  |  outcome=1: bugfix/security commit touched PR files)",
             fontsize=11, fontweight="bold")

model_keys = ["qwen2.5:7b-instruct", "llama3.1:8b", "gemma3:4b"]
model_names = ["Qwen2.5-7B", "Llama3.1-8B", "Gemma3-4B"]

for ax, mkey, mname in zip(axes, model_keys, model_names):
    model_results = [r for r in rq1["results"]
                     if r.get("model_name") == mkey
                     and isinstance(r.get("risk_score"), int)
                     and r.get("pr_id") in outcome_map]
    
    scores_0 = [r["risk_score"] for r in model_results if outcome_map[r["pr_id"]] == 0]
    scores_1 = [r["risk_score"] for r in model_results if outcome_map[r["pr_id"]] == 1]
    
    bins = [0.5, 1.5, 2.5, 3.5, 4.5, 5.5]
    ax.hist(scores_0, bins=bins, alpha=0.7, color=TEAL,  label=f"No outcome (n={len(scores_0)})",
            density=True, zorder=3)
    ax.hist(scores_1, bins=bins, alpha=0.7, color=RED,   label=f"Outcome=1 (n={len(scores_1)})",
            density=True, zorder=3)
    
    mean_0 = np.mean(scores_0) if scores_0 else 0
    mean_1 = np.mean(scores_1) if scores_1 else 0
    ax.axvline(mean_0, color=TEAL, linestyle="--", linewidth=1.5)
    ax.axvline(mean_1, color=RED,  linestyle="--", linewidth=1.5)
    
    ax.set_title(f"{mname}\nμ(0)={mean_0:.2f}  μ(1)={mean_1:.2f}",
                 fontsize=11, fontweight="bold")
    ax.set_xlabel("Risk Score (1–5)")
    ax.set_ylabel("Density" if ax == axes[0] else "")
    ax.set_xticks([1,2,3,4,5])
    ax.legend(fontsize=9)
    ax.yaxis.grid(True, color=LGRAY, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.spines[["top","right"]].set_visible(False)

plt.tight_layout()
save(fig, "../figures/rq3/rq3_risk_by_outcome.png")