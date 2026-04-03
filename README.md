# PR Review Effort Predictor

## Research Question

**RQ1: How can we determine which pull requests will need extensive review work?**

This project uses small, locally-hosted LLMs as "judges" to predict which pull requests in a large open-source repository (pandas-dev/pandas) will require high review effort and compares them against a Random Forest baseline using McNemar's test.

---

## Project Overview

The pipeline has three stages:

1. **Data Collection** - Fetch 500 merged PRs from GitHub, compute structural and AST-based code complexity features, and store everything in `pr_dataset.json`.
2. **LLM Judging & Evaluation** - Three local LLMs score each PR's review effort on a 1-5 rubric. Predictions are evaluated against a ground-truth effort label derived from review duration and comment count.
3. **Visualization** - Generate plots comparing model performance, risk score distributions, circularity checks, and McNemar contingency tables.

---

## Repository Structure

| File | Purpose |
|------|---------|
| `config.py` | GitHub API token and base URL configuration |
| `get_pull_request_data.py` | Fetches PR metadata and diffs from GitHub API, computes AST features → `pr_dataset.json` |
| `get_cyclomatic_complexity.py` | Calculates total cyclomatic complexity of Python source using Radon |
| `get_nesting_depth.py` | Computes maximum nesting depth of control-flow structures via AST |
| `get_logic_density.py` | Measures ratio of logic lines (if/for/while/try/return) to total lines |
| `llm_judge_rq1.py` | Main evaluation script, runs 3 LLM judges, computes metrics, RF baseline, McNemar test |
| `get_baseline_model.py` | Random Forest baseline (80/20 stratified split) on structural + semantic features |
| `get_spearman_correlation.py` | Circularity check : Spearman correlation between input features and ground-truth effort |
| `generate_plots.py` | Produces all result visualizations from `llm_judge_results.json` |
| `pr_dataset.json` | Collected PR dataset (500 PRs) |
| `llm_judge_results.json` | Full output: per-PR scores, rationales, and aggregate metrics |

---

## Setup

### 1. Install Python dependencies

```bash
pip install requests radon scikit-learn numpy scipy statsmodels matplotlib
```

### 2. Install Ollama

Download and install from [https://ollama.com/download](https://ollama.com/download). Ollama starts automatically after installation.

### 3. Pull the models

```bash
ollama pull qwen2.5:7b-instruct
ollama pull llama3.1:8b
ollama pull gemma3:4b
```

### 4. Configure GitHub API token

Create a `config.py` file in the project root:

```python
config = {
    "API_TOKEN": "your-github-token-here",
    "BASE_URL": "https://api.github.com",
}
```

A personal access token with public repo read access is sufficient.

---

## How to Run

```bash
# Step 1 - Collect PR data from GitHub (skip if pr_dataset.json already exists)
python get_pull_request_data.py

# Step 2 - Run the LLM judges and evaluation (make sure Ollama is running)
python llm_judge_rq1.py

# Step 3 - Generate result plots
python generate_plots.py
```

- Step 1 fetches up to 1500 raw PRs and filters down to 500 valid merged PRs. It checkpoints progress to `pr_dataset_checkpoint.json` every 100 PRs.
- Step 2 writes all scores, rationales, and metrics to `llm_judge_results.json`.
- Step 3 reads from `llm_judge_results.json` and produces four plots.

---

## Methodology

### Ground Truth Labeling

Each PR receives a continuous effort score combining two features:

```
effort_score = 0.6 × normalized_review_duration + 0.4 × normalized_comment_count
```

Both values are min-max normalized to [0, 1] across the full dataset. PRs at or above the **75th percentile** are labeled **high effort (1)**; the rest are **low effort (0)**.

### LLM-J Prediction

Each LLM scores PRs on a 1–5 rubric focused on functional complexity (not raw patch size). A PR is predicted **high effort** if the LLM assigns a risk score ≥ 4.

The LLM receives: PR title, description (truncated), structural metrics (additions, deletions, files changed, commits), functional metrics (cyclomatic delta, max nesting depth, logic density), and the first 800 characters of the diff.

### Random Forest Baseline

A Random Forest classifier (100 trees, 80/20 stratified split) trained on four features: cyclomatic delta, max nesting depth, logic density, and total patch size (additions + deletions).

### Statistical Comparison

McNemar's test compares disagreement patterns between the LLM judge and RF baseline on the test set.

### Circularity Check

Spearman rank correlation between each input feature and the ground-truth effort score, to verify that the LLM is not simply showing features already present in the label.

---

## Results (500 PRs from pandas-dev/pandas)

### Model Performance (High-Effort Class)

| Model | Precision | Recall | F1 | Avg Risk Score |
|-------|-----------|--------|----|----------------|
| Qwen2.5-7B | 0.80 | 0.03 | 0.06 | 1.90 / 5 |
| Llama3.1-8B | 0.49 | 0.14 | 0.21 | 2.30 / 5 |
| Gemma3-4B | 0.30 | 0.56 | 0.39 | 3.26 / 5 |
| RF Baseline | 0.20 | 0.12 | 0.15 | - |

### McNemar Test (LLM-J vs RF Baseline)

| Model | b (LLM wrong, RF right) | c (LLM right, RF wrong) | p-value | Significant (α=0.05) |
|-------|-------------------------|-------------------------|---------|-----------------------|
| Qwen2.5-7B | 12 | 3 | 0.035 | Yes |
| Llama3.1-8B | 13 | 3 | 0.021 | Yes |
| Gemma3-4B | 13 | 17 | 0.585 | No |

### Circularity Check

| Feature | Spearman ρ | Significant |
|---------|-----------|-------------|
| Additions | 0.233 | Yes |
| Deletions | 0.096 | Yes |
| Cyclomatic Δ | 0.155 | Yes |
| Max Nesting | 0.114 | Yes |
| Logic Density | 0.032 | No |

Most input features show weak-to-moderate correlation with the effort label. Logic density is the only feature with no significant circularity risk.

### Key Takeaways

- **Gemma3-4B** achieves the best F1 (0.39) with the highest recall (0.56), making it the most useful model for flagging high-effort PRs, though at the cost of lower precision.
- **Qwen2.5-7B** has the highest precision (0.80) but almost never predicts high effort (recall = 0.03), making it too conservative for practical use.
- **All three LLMs outperform the RF baseline on F1**, though Qwen and Llama achieve this through statistically different disagreement patterns (significant McNemar p-values), while Gemma's improvement is not statistically significant.

---

## Generated Plots

| Plot | Description |
|------|-------------|
| `metrics.png` | Precision, Recall, and F1 grouped bar chart by model |
| `circularity_check.png` | Spearman ρ for each feature vs ground-truth effort |
| `mcnemar_results.png` | McNemar contingency tables (LLM-J vs RF) per model |
| `risk_distribution.png` | Risk score distribution histograms per model |

---

## Models Used

| Model | Parameters | Inference |
|-------|-----------|-----------|
| Qwen2.5-7B Instruct | 7B | Local via Ollama |
| Llama3.1-8B | 8B | Local via Ollama |
| Gemma3-4B | 4B | Local via Ollama |

All inference is performed locally - no API keys or cloud credits required beyond the GitHub token for data collection.
