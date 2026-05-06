# Optimizing Reviewer Effort through Functional Logic in OSS

LLM-as-a-Judge evaluation across three research questions on 500 PRs from `django/django`. RQ1 runs locally via Ollama. RQ2 and RQ3 run on Google Colab via HuggingFace. Statistical tests include McNemar's test and Spearman rank correlation.

---

## Pre-computed Artifacts

This repository ships the artifacts referenced in the paper, so the numbers can be verified without re-running the full pipeline:

- `data/pr_dataset_django.json` collected and filtered PRs
- `data/rq3_outcome_labels.json` ground-truth labels for RQ3
- `results/llm_judge_results_django.json` per-PR risk scores and rationales (RQ1)
- `results/llm_judge_rq2_results.json` segment-level results (RQ2)
- `results/llm_judge_rq3_results.json` post-merge correlation outputs (RQ3)
- `results/segment_dataset.json` extracted diff hunks used for RQ2
- `figures/rq1/`, `figures/rq2/`, `figures/rq3/` paper figures

To regenerate any of these from scratch, follow the "Full Reproduction" section below.

---

## Setup

### 1. Install dependencies

```bash
pip install requests radon scikit-learn numpy scipy statsmodels matplotlib
```

Tested on Python 3.10 and 3.11.

### 2. Install Ollama and pull the three models (RQ1 only)

Ollama is required only for RQ1. RQ2 and RQ3 run on Google Colab and do not need Ollama.

Download Ollama from <https://ollama.com/download>, then:

```bash
ollama pull qwen2.5:7b-instruct
ollama pull llama3.1:8b
ollama pull gemma3:4b
```

Make sure `ollama serve` is running in the background before launching any RQ1 command.

### 3. Add a GitHub token

Create `src/config.py` with the following contents:

```python
config = {
    "API_TOKEN": "your-github-token-here",
    "BASE_URL": "https://api.github.com",
}
```

Generate a personal access token at <https://github.com/settings/tokens> with `public_repo` scope. The token is only needed for fetching PR data; if `data/pr_dataset_django.json` is already present, you can skip data collection and the token is not required.

---

## Full Reproduction

All commands run from the project root.

### RQ1: PR-level effort prediction (local, Ollama)

```bash
# Step 1: collect PR data
python -m src.get_pull_request_data --repository django/django --out_file data/pr_dataset_django.json

# Step 2: run the LLM judge
python -m evaluation.llm_judge_rq1 --dataset data/pr_dataset_django.json --out results/llm_judge_results_django.json

# Step 3: generate plots
python -m evaluation.generate_plots --rq1 results/llm_judge_results_django.json --repo django/django --outdir figures/rq1
```

### RQ2: Segment-level comment prediction (Colab, GPU)

Open `notebooks/RQ2_LLM_Judge.ipynb` in Google Colab. Upload `data/pr_dataset_django.json` when prompted. Run all cells. Download `results/llm_judge_rq2_results.json` when complete.

RQ2 requires two Colab Secrets:

- `HF_TOKEN`: a HuggingFace token with access to Llama 3.1 (a gated model)
- `GITHUB_TOKEN`: a GitHub personal access token with `public_repo` scope, used to fetch segment-level inline reviewer comments via the GitHub API

### RQ3: Post-merge outcome correlation (Colab)

Open `notebooks/RQ3_LLM_Judge.ipynb` in Google Colab. Upload `data/pr_dataset_django.json` and `results/llm_judge_results_django.json` when prompted. Run all cells. Download `results/llm_judge_rq3_results.json` when complete.

After downloading the RQ3 results, generate plots locally:

```bash
python -m evaluation.get_plots_rq3
```

---

## Repository Structure

```
.
├── src/                   data collection and feature extraction
├── evaluation/            evaluation scripts and plot generation
├── notebooks/             RQ2 and RQ3 Colab notebooks
├── data/                  collected PR datasets
├── results/               LLM judge outputs and metrics
└── figures/               generated plots (rq1/, rq2/, rq3/)
```

---

## Notes

- The pre-computed JSONs and figures listed at the top let a reviewer verify our results in seconds. The "Full Reproduction" steps regenerate them from scratch.
- `src/get_pull_request_data.py` checkpoints every 100 PRs and supports resume, so it is safe to interrupt and restart.
- GitHub's secondary rate limit can trigger around PR 80 to 100 on commit-history queries. The RQ3 notebook handles this with `Retry-After` parsing; for RQ1 collection, the script sleeps and retries with exponential backoff.
- LLM temperature is fixed at `0.1` throughout for consistent scoring.
- The `--limit N` flag on `get_pull_request_data.py` and `llm_judge_rq1.py` truncates runs to `N` PRs, useful for smoke tests.