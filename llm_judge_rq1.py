import json
import requests
import statistics
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report
from get_baseline_model import compute_rf_baseline
from statsmodels.stats.contingency_tables import mcnemar as mcnemar_test
from get_spearman_correlation import check_circularity


SYSTEM_PROMPT = """You are an expert software engineering reviewer.
Your task is to assess how much review effort a pull request will require,
based on its structural and functional complexity metrics.

You will be given:
  - PR title and description
  - Code diff (truncated)
  - Structural metrics: additions, deletions, files_changed, commit_count
  - Functional metrics: cyclomatic_delta_total, max_nesting_depth, logic_density_total

Scoring criteria (point-wise, 1-5):
  1 = Trivial. Minimal logic change, easy to review quickly.
  2 = Simple. Slight complexity increase but straightforward.
  3 = Moderate. Some logic changes that warrant careful reading.
  4 = High effort. Complex control flow, deep nesting, or high logic density.
  5 = Very high effort. Major functional restructuring; needs thorough review.

Important: Focus on FUNCTIONAL complexity, not patch size alone.
A 500-line diff of docstrings is less effort than a 30-line change to core control flow.

Respond in this exact JSON format (no markdown, no code fences):
{
  "risk_score": <integer 1-5>,
  "rationale": "<one sentence explaining the key driver of this score>"
}"""


def build_user_message(pr: dict) -> str:
    diff_snippet = (pr.get("diff") or "")[:800]
    return f"""PR Title: {pr.get('title', 'N/A')}

Description: {(pr.get('body') or 'No description')[:300]}

Structural metrics:
  additions={pr.get('additions', 0)}, deletions={pr.get('deletions', 0)},
  files_changed={pr.get('files_changed', 0)}, commits={pr.get('commit_count', 0)}

Functional metrics:
  cyclomatic_delta_total={pr.get('cyclomatic_delta_total', 0)},
  max_nesting_depth={pr.get('max_nesting_depth', 0)},
  logic_density_total={pr.get('logic_density_total', 0.0):.3f}

Diff (first 800 chars):
{diff_snippet}"""


MODELS = [
    "qwen2.5:7b-instruct",
    "llama3.1:8b",
    "gemma3:4b",
]


def clean_raw_response(raw: str) -> str:
    """
    Clean LLM response to extract pure JSON.
    Handles:
      - <think>...</think> tags (Qwen reasoning models)
      - ```json ... ``` markdown fences (Gemma, Llama)
      - ``` ... ``` plain fences
      - Leading/trailing whitespace
    """
    # strip thinking tags
    if "<think>" in raw:
        raw = raw[raw.rfind("</think>") + len("</think>"):].strip()

    # strip markdown code fences
    if "```" in raw:
        parts = raw.split("```")
        for part in parts:
            part = part.strip()
            if part.startswith("json"):
                part = part[4:].strip()
            if part.startswith("{"):
                raw = part
                break

    return raw.strip()


def call_llm_judge(pr: dict, model_name: str) -> dict:
    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_message(pr)}
        ],
        "stream": False,
        "options": {"temperature": 0.1}
    }

    for attempt in range(3):
        try:
            resp = requests.post(
                "http://localhost:11434/api/chat",
                json=payload,
                timeout=120
            )
            raw = resp.json()["message"]["content"].strip()
            raw = clean_raw_response(raw)

            try:
                return json.loads(raw)
            except json.JSONDecodeError:
                if attempt == 2:
                    print(f"  Failed to parse JSON for {model_name}: {raw[:100]}")
                    return {"risk_score": None, "rationale": raw}
                continue

        except Exception as e:
            print(f"  Attempt {attempt+1}/3 failed for {model_name}: {e}")
            if attempt < 2:
                import time
                time.sleep(5)
            else:
                return {"risk_score": None, "rationale": str(e)}


def compute_effort_scores(records: list) -> list:
    """
    Effort score = 0.6 x normalized_review_duration + 0.4 x normalized_comment_count
    Both signals normalized to [0,1] independently via min-max.
    """
    durations = [r.get("review_duration_hours", 0) for r in records]
    comments  = [r.get("total_comments", 0) for r in records]

    def normalize(values):
        mn, mx = min(values), max(values)
        if mx == mn:
            return [0.0] * len(values)
        return [(v - mn) / (mx - mn) for v in values]

    norm_dur  = normalize(durations)
    norm_comm = normalize(comments)
    return [0.6 * d + 0.4 * c for d, c in zip(norm_dur, norm_comm)]


def run_mcnemar(llm_preds: list, rf_preds: list, gt: list) -> tuple:
    """
    McNemar test comparing LLM-J vs RF baseline on the same test PRs.
    Returns (p_value, contingency_table_dict)
    """
    a = sum(1 for l, r, g in zip(llm_preds, rf_preds, gt) if l == g and r == g)
    b = sum(1 for l, r, g in zip(llm_preds, rf_preds, gt) if l != g and r == g)
    c = sum(1 for l, r, g in zip(llm_preds, rf_preds, gt) if l == g and r != g)
    d = sum(1 for l, r, g in zip(llm_preds, rf_preds, gt) if l != g and r != g)

    print(f"  McNemar table: both_correct={a}, llm_wrong_rf_right={b}, "
          f"llm_right_rf_wrong={c}, both_wrong={d}")
    print(f"  Disagreements (b+c): {b+c}")

    table = np.array([[a, b], [c, d]])
    result = mcnemar_test(table, exact=True)
    return result.pvalue, {"a": a, "b": b, "c": c, "d": d}


def evaluate(input_path: str = "pr_dataset.json",
             output_path: str = "llm_judge_results.json"):

    with open(input_path, encoding="utf-8") as f:
        dataset = json.load(f)

    # compute ground truth ONCE from full dataset
    # ensures all models are compared against identical labels
    effort_values_all = compute_effort_scores(dataset)
    effort_q3 = np.percentile(effort_values_all, 75)
    gt_lookup = {
        pr.get("pr_id"): 1 if effort_values_all[i] >= effort_q3 else 0
        for i, pr in enumerate(dataset)
    }

    print(f"Dataset size: {len(dataset)} PRs")
    print(f"High effort threshold (Q75): {effort_q3:.4f}")
    high_effort_count = sum(gt_lookup.values())
    print(f"High effort PRs: {high_effort_count} ({high_effort_count/len(dataset)*100:.1f}%)")

    # run LLM judge for all models
    results = []
    for i, pr in enumerate(dataset):
        print(f"Judging PR {i+1}/{len(dataset)}: id={pr.get('pr_id', '?')}")
        for model_name in MODELS:
            verdict = call_llm_judge(pr, model_name)
            results.append({
                "pr_id":                 pr.get("pr_id"),
                "title":                 pr.get("title"),
                "risk_score":            verdict.get("risk_score"),
                "rationale":             verdict.get("rationale"),
                "review_duration_hours": pr.get("review_duration_hours", 0),
                "total_comments":        pr.get("total_comments", 0),
                "cyclomatic_delta":      pr.get("cyclomatic_delta_total", 0),
                "max_nesting":           pr.get("max_nesting_depth", 0),
                "logic_density":         pr.get("logic_density_total", 0.0),
                "additions":             pr.get("additions", 0),
                "deletions":             pr.get("deletions", 0),
                "model_name":            model_name,
            })

    # evaluate each model using shared ground truth
    all_metrics = {}

    for model_name in MODELS:
        print(f"\n{'='*50}")
        print(f"Evaluating: {model_name}")
        print(f"{'='*50}")

        model_data = [r for r in results if r["model_name"] == model_name]
        valid = [r for r in model_data if isinstance(r["risk_score"], int)]

        # debug parse rate
        print(f"  Total results    : {len(model_data)}")
        print(f"  Valid (parsed)   : {len(valid)}")
        print(f"  Failed to parse  : {len(model_data) - len(valid)}")
        if len(model_data) - len(valid) > 0:
            failed = [r for r in model_data if not isinstance(r["risk_score"], int)]
            print(f"  Example failure  : {failed[0]['rationale'][:200]}")

        if len(valid) < 10:
            print(f"  Not enough valid PRs for evaluation.")
            all_metrics[model_name] = {
                "note": "Not enough valid PRs.",
                "total": len(model_data),
                "valid": len(valid),
                "failed": len(model_data) - len(valid)
            }
            continue

        # use shared ground truth labels
        gt_labels = [gt_lookup[r["pr_id"]] for r in valid]
        effort_values = [effort_values_all[i]
                         for i, pr in enumerate(dataset)
                         if pr.get("pr_id") in {r["pr_id"] for r in valid}]

        # circularity check
        circularity_results = check_circularity(valid, effort_values)

        # LLM-J predictions using fixed rubric threshold
        score_values = [r["risk_score"] for r in valid]
        llm_pred_all = [1 if s >= 4 else 0 for s in score_values]

        # RF baseline — 80/20 stratified split
        rf_preds, y_test, idx_test = compute_rf_baseline(valid, gt_labels)
        llm_preds_test = [llm_pred_all[i] for i in idx_test]

        # McNemar test on shared test set
        p_mcnemar, mcnemar_table = run_mcnemar(
            llm_preds_test, rf_preds.tolist(), y_test.tolist()
        )

        # metrics for LLM-J on full dataset
        precision_llm = precision_score(gt_labels, llm_pred_all, zero_division=0)
        recall_llm    = recall_score(gt_labels, llm_pred_all, zero_division=0)
        f1_llm        = f1_score(gt_labels, llm_pred_all, zero_division=0)

        # metrics for RF on test set
        precision_rf = precision_score(y_test, rf_preds, zero_division=0)
        recall_rf    = recall_score(y_test, rf_preds, zero_division=0)
        f1_rf        = f1_score(y_test, rf_preds, zero_division=0)

        all_metrics[model_name] = {
            "rq1": "Which PRs will need extensive review work?",
            "technique": "Point-wise LLM-J with structured prompting",
            "n_prs_evaluated":       len(valid),
            "n_prs_failed_parse":    len(model_data) - len(valid),
            "high_effort_threshold": round(effort_q3, 2),
            "llm_j": {
                "precision":      round(precision_llm, 4),
                "recall":         round(recall_llm, 4),
                "f1":             round(f1_llm, 4),
                "avg_risk_score": round(statistics.mean(score_values), 2),
            },
            "random_forest_baseline": {
                "precision": round(precision_rf, 4),
                "recall":    round(recall_rf, 4),
                "f1":        round(f1_rf, 4),
                "test_size": len(idx_test),
            },
            "mcnemar_p_value":   round(p_mcnemar, 4),
            "mcnemar_table":     mcnemar_table,
            "circularity_check": circularity_results,
        }

        print(f"\n  {'Model':<20} {'Precision':>10} {'Recall':>10} {'F1':>10}")
        print(f"  {'-'*52}")
        print(f"  {'LLM-J':<20} {precision_llm:>10.4f} {recall_llm:>10.4f} {f1_llm:>10.4f}")
        print(f"  {'RF Baseline':<20} {precision_rf:>10.4f} {recall_rf:>10.4f} {f1_rf:>10.4f}")
        print(f"\n  Avg LLM risk score : {all_metrics[model_name]['llm_j']['avg_risk_score']} / 5")
        print(f"  McNemar p-value    : {p_mcnemar:.4f} "
              f"({'significant' if p_mcnemar < 0.05 else 'not significant'} at alpha=0.05)")
        print("\n  LLM-J Classification Report:")
        print(classification_report(gt_labels, llm_pred_all,
                                    target_names=["low effort", "high effort"],
                                    zero_division=0))

    output = {"metrics": all_metrics, "results": results}
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\nFull results written to {output_path}")


if __name__ == "__main__":
    evaluate()