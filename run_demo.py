
import subprocess
import sys


STEPS = [
    {
        "label": "Step 1/3: Fetching 10 PRs from django/django",
        "cmd": [
            sys.executable, "-m", "src.get_pull_request_data",
            "--repository",      "django/django",
            "--out_file",        "demo/data/pr_dataset_django_live.json",
            "--checkpoint_file", "demo/data/pr_dataset_django_live_ckpt.json",
            "--limit",           "10",
        ],
    },
    {
        "label": "Step 2/3: Running LLM judge with Gemma3-4B",
        "cmd": [
            sys.executable, "-m", "evaluation.llm_judge_rq1",
            "--dataset", "demo/data/pr_dataset_django_live.json",
            "--out",     "demo/results/demo_rq1_django_live.json",
            "--models",  "gemma3:4b",
        ],
    },
    {
        "label": "Step 3/3: Generating plots",
        "cmd": [
            sys.executable, "-m", "evaluation.generate_plots",
            "--rq1",    "demo/results/demo_rq1_django_live.json",
            "--repo",   "django/django",
            "--outdir", "demo/figures",
        ],
    },
]


def main():
    for step in STEPS:
        print(f"  {step['label']}")
        result = subprocess.run(step["cmd"])
        if result.returncode != 0:
            print()
            print(f"FAILED at: {step['label']} (exit code {result.returncode})")
            sys.exit(result.returncode)


    print("Demo complete.")
    print("Dataset : demo/data/pr_dataset_django_live.json")
    print("Results : demo/results/demo_rq1_django_live.json")
    print("Plots   : demo/figures/rq1/")


if __name__ == "__main__":
    main()