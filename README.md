# Optimizing Reviewer Effort through Functional Logic in OSS

LLM-as-a-Judge evaluation across three research questions on 500 PRs each from `pandas-dev/pandas` and `django/django`. Models run locally via Ollama (RQ1) and HuggingFace on Google Colab (RQ2). Statistical tests include McNemar's test and Spearman rank correlation.

---

## Setup

### 1. Install dependencies

```bash
pip install requests radon scikit-learn numpy scipy statsmodels matplotlib
```

### 2. Install Ollama and pull models

Download from [https://ollama.com/download](https://ollama.com/download), then:

```bash
ollama pull qwen2.5:7b-instruct
ollama pull llama3.1:8b
ollama pull gemma3:4b
```

### 3. Add GitHub token

```python
config = {
    "API_TOKEN": "your-github-token-here",
    "BASE_URL": "https://api.github.com",
}
```

---

## How to Run

All commands run from the **project root**.

### RQ1 — PR-level effort prediction (local)

```bash
python -m src.get_pull_request_data --repository pandas-dev/pandas --out_file data/pr_dataset_pandas.json
python -m src.get_pull_request_data --repository django/django --out_file data/pr_dataset_django.json

python -m evaluation.llm_judge_rq1 --dataset data/pr_dataset_pandas.json --out results/llm_judge_results.json
python -m evaluation.llm_judge_rq1 --dataset data/pr_dataset_django.json --out results/llm_judge_results_django.json

python -m evaluation.plot_results --rq1 results/llm_judge_results.json --repo pandas-dev/pandas --outdir figures/rq1
python -m evaluation.plot_results --rq1 results/llm_judge_results_django.json --repo django/django --outdir figures/rq1
```

### RQ2 — Segment-level comment prediction (Google Colab, GPU)

Open `notebooks/RQ2_LLM_Judge.ipynb` in Colab with a T4 GPU runtime. Upload `data/pr_dataset_django.json` when prompted. Download `results/llm_judge_rq2_results.json` when complete.

Requires a HuggingFace token with access to gated models (Llama 3.1). Add it to Colab Secrets as `HF_TOKEN`.

### RQ3 — Post-merge outcome correlation (Google Colab, CPU)

Open `notebooks/RQ3_LLM_Judge.ipynb` in Colab with a CPU runtime. Upload `data/pr_dataset_django.json` and `results/llm_judge_results_django.json` when prompted. Download `results/llm_judge_rq3_results.json` when complete.

```bash
python -m evaluation.get_plots_rq3
```

---

## Repository Structure

```
.
├── src/                          # Data collection and feature extraction
├── evaluation/                   # Evaluation scripts and plot generation
├── notebooks/                    # RQ2 and RQ3 Colab notebooks
├── data/                         # Collected PR datasets
├── results/                      # LLM judge outputs and metrics
└── figures/                      # Generated plots (rq1/, rq2/, rq3/)
```

---

## Notes

- If `data/` files are already present, skip the collection step entirely.
- Data collection checkpoints every 50 PRs — safe to interrupt and resume.
- GitHub's secondary rate limit triggers around PR 80–100 on commit history queries. The RQ3 notebook handles this automatically with `Retry-After` header parsing.
- LLM temperature is `0.1` throughout for consistent scoring.
