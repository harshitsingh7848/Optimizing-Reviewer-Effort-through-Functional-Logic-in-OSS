# Optimizing Reviewer Effort through Functional Logic in OSS

LLM-as-a-Judge evaluation across three research questions on 500 PRs from `django/django`. RQ1 runs locally via Ollama. RQ2 and RQ3 run on Google Colab via HuggingFace. Statistical tests include McNemar's test and Spearman rank correlation.

---

## Quick Start (Demo)

If you just want to verify the pipeline runs end-to-end, run the demo. It fetches 10 PRs, scores them with Gemma3-4B, and produces plots in about 3 minutes.

```bash
python run_demo.py
```

Outputs land in `demo/data/`, `demo/results/`, and `demo/figures_live/`.

For full reproduction of the paper results, follow the sections below.

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

Create `src/config.py` (or edit it if it already exists) with the following contents:

```python
config = {
    "API_TOKEN": "your-github-token-here",
    "BASE_URL": "https://api.github.com",
}
```

Generate a personal access token at <https://github.com/settings/tokens> with `public_repo` scope. The token is only needed for fetching PR data; if `data/` already contains the pre-collected dataset, you can skip data collection and the token is not required.

---

## How to Run (Full Reproduction)

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

RQ2 requires a HuggingFace token with access to Llama 3.1 (a gated model). Add it to Colab Secrets as `HF_TOKEN`.

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
├── run_demo.py            quick-start wrapper for RQ1
├── src/                   data collection and feature extraction
├── evaluation/            evaluation scripts and plot generation
├── notebooks/             RQ2 and RQ3 Colab notebooks
├── data/                  collected PR datasets
├── results/               LLM judge outputs and metrics
└── figures/               generated plots (rq1/, rq2/, rq3/)
```

---

## Notes

- If `data/` already contains the JSON dataset, skip the collection step in RQ1. Data collection is the slowest part of the pipeline.
- `src/get_pull_request_data.py` checkpoints every 100 PRs and supports resume, so it is safe to interrupt and restart.
- GitHub's secondary rate limit can trigger around PR 80 to 100 on commit-history queries. The RQ3 notebook handles this with `Retry-After` parsing; for RQ1 collection, the script sleeps and retries with exponential backoff.
- LLM temperature is fixed at `0.1` throughout for consistent scoring.
- The `--limit N` flag on `get_pull_request_data.py` and `llm_judge_rq1.py` truncates runs to `N` PRs, useful for smoke tests.