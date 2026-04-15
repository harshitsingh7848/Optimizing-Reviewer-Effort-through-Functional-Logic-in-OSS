import requests
from src.config import config
import datetime
import json
import base64
import time
from src.get_cyclomatic_complexity import cyclomatic_complexity_total
from src.get_nesting_depth import max_nesting_depth
from src.get_logic_density import logic_density
import argparse

HEADERS = {
    "Accept": "application/vnd.github+json",
    "Authorization": f'Bearer {config["API_TOKEN"]}',
    "X-GitHub-Api-Version": "2022-11-28",
}

TARGET_VALID_PRS  = 500
CHECKPOINT_EVERY  = 100
RATE_LIMIT_BUFFER = 100


def get_pr_page(repository_path: str, page: int) -> list:
    """Fetch a single page of 100 closed PRs."""
    url = f'{config["BASE_URL"]}/repos/{repository_path}/pulls'
    params = {"state": "closed", "per_page": 100, "page": page}
    try:
        response = requests.get(url, headers=HEADERS, params=params, timeout=30)
        response.raise_for_status()
        return response.json()
    except Exception as error:
        print(f"  Error fetching page {page}: {error}")
        return []


def get_pr_details(pr_number, repository_path, retries=3):
    url = f'{config["BASE_URL"]}/repos/{repository_path}/pulls/{pr_number}'
    for attempt in range(retries):
        try:
            response = requests.get(url, headers=HEADERS, timeout=30)
            response.raise_for_status()
            return response.json()
        except Exception as error:
            print(f"Attempt {attempt+1}/{retries} failed for PR details {pr_number}: {error}")
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
    return {}


def get_pr_files(pr_number, repository_path, retries=3):
    url = f'{config["BASE_URL"]}/repos/{repository_path}/pulls/{pr_number}/files'
    for attempt in range(retries):
        try:
            response = requests.get(url, headers=HEADERS, timeout=30)
            response.raise_for_status()
            return response.json()
        except Exception as error:
            print(f"Attempt {attempt+1}/{retries} failed for PR files {pr_number}: {error}")
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
    return []


def get_file_content_at_ref(file_path, ref, repository_path, retries=3):
    url = f'{config["BASE_URL"]}/repos/{repository_path}/contents/{file_path}'
    params = {"ref": ref}
    for attempt in range(retries):
        try:
            response = requests.get(url, headers=HEADERS, params=params, timeout=30)
            if response.status_code == 404:
                return None
            response.raise_for_status()
            data = response.json()
            if isinstance(data, list):
                return None
            content  = data.get("content", "")
            encoding = data.get("encoding", "")
            if encoding == "base64":
                return base64.b64decode(content).decode("utf-8", errors="ignore")
            return None
        except Exception as error:
            print(f"Attempt {attempt+1}/{retries} failed for {file_path}: {error}")
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
    return None


def check_rate_limit():
    url = f'{config["BASE_URL"]}/rate_limit'
    resp = requests.get(url, headers=HEADERS).json()
    remaining = resp["rate"]["remaining"]
    reset_time = resp["rate"]["reset"]
    if remaining < RATE_LIMIT_BUFFER:
        wait = reset_time - time.time() + 10
        print(f"Rate limit low ({remaining} remaining). Waiting {wait:.0f}s...")
        time.sleep(max(wait, 0))
    else:
        print(f"Rate limit: {remaining} remaining")



MIN_COMMITS        = 2
BOT_PATTERNS       = ["bot", "dependabot", "github-actions",
                       "renovate", "allcontributors", "codecov"]
CODE_EXTENSIONS    = {".py", ".c", ".cpp", ".java", ".js", ".ts"}
DEPENDENCY_KEYWORDS = ["bump", "update dependency", "upgrade", "dependabot"]


def is_bot_pr(pr):
    login = (pr.get("user") or {}).get("login", "").lower()
    return any(p in login for p in BOT_PATTERNS)

def is_doc_only_pr(files):
    return all(
        not any(f.get("filename", "").endswith(ext) for ext in CODE_EXTENSIONS)
        for f in files
    )

def is_dependency_only_pr(pr):
    return any(kw in pr.get("title", "").lower() for kw in DEPENDENCY_KEYWORDS)

def is_empty_diff_pr(files):
    return not files or all(f.get("changes", 0) == 0 for f in files)

def is_low_commit_pr(details):
    return details.get("commits", 0) < MIN_COMMITS


def compute_review_time(created, merged):
    if not merged:
        return 0
    fmt = "%Y-%m-%dT%H:%M:%SZ"
    t1 = datetime.datetime.strptime(created, fmt)
    t2 = datetime.datetime.strptime(merged, fmt)
    return (t2 - t1).total_seconds() / 3600


def compute_file_features(code):
    if not code or not code.strip():
        return {"cyclomatic": 0, "nesting": 0, "logic_density": 0.0}
    return {
        "cyclomatic":    cyclomatic_complexity_total(code) or 0,
        "nesting":       max_nesting_depth(code) or 0,
        "logic_density": logic_density(code) or 0.0,
    }


def compute_pr_ast_features(details, files, repository_path):
    base_sha = details["base"]["sha"]
    head_sha = details["head"]["sha"]

    cyclomatic_delta = 0
    max_nest         = 0
    ld_values        = []

    for f in files:
        path = f.get("filename", "")
        if not path.endswith(".py"):
            continue

        before = get_file_content_at_ref(path, base_sha, repository_path)
        after  = get_file_content_at_ref(path, head_sha, repository_path)

        bf = compute_file_features(before or "")
        af = compute_file_features(after  or "")

        cyclomatic_delta += af["cyclomatic"] - bf["cyclomatic"]
        max_nest = max(max_nest, af["nesting"])
        ld_values.append(af["logic_density"])

    return {
        "cyclomatic_delta_total": cyclomatic_delta,
        "max_nesting_depth":      max_nest,
        "logic_density_total":    sum(ld_values) / len(ld_values) if ld_values else 0.0,
    }


def main(repository_path: str, out_file_path: str, checkpoint_file_path: str):
    try:
        with open(checkpoint_file_path, encoding="utf-8") as f:
            results = json.load(f)
        done_ids = {r["pr_id"] for r in results}
        print(f"Resuming: {len(results)} PRs already collected")
    except FileNotFoundError:
        results, done_ids = [], set()

    skipped      = 0
    page         = 1
    raw_seen     = 0

    print(f"Collecting {TARGET_VALID_PRS} valid PRs from {repository_path}\n")

    while len(results) < TARGET_VALID_PRS:
        batch = get_pr_page(repository_path, page)

        if not batch:
            print(f"\nNo more PRs available after page {page - 1} "
                  f"({raw_seen} raw PRs seen).")
            break

        print(f"Page {page}: fetched {len(batch)} PRs "
              f"(valid so far: {len(results)}/{TARGET_VALID_PRS})")
        raw_seen += len(batch)
        page     += 1
        time.sleep(0.5)

        for pr in batch:
            if len(results) >= TARGET_VALID_PRS:
                break

            pr_number = pr["number"]

            if pr["id"] in done_ids:
                continue
            if pr.get("draft", False):
                skipped += 1; continue
            if is_bot_pr(pr):
                skipped += 1; continue
            if is_dependency_only_pr(pr):
                skipped += 1; continue

            details = get_pr_details(pr_number, repository_path)
            files   = get_pr_files(pr_number, repository_path)

            if not details.get("merged_at"):
                skipped += 1; continue
            if is_empty_diff_pr(files):
                skipped += 1; continue
            if is_doc_only_pr(files):
                skipped += 1; continue


            print(f"Valid PR {len(results) + 1}/{TARGET_VALID_PRS}: #{pr_number}")

            diff_text = "\n".join(
                f"--- a/{f.get('filename','')}\n+++ b/{f.get('filename','')}\n{f.get('patch', '')}"
                for f in files if "patch" in f
            )

            ast_features = compute_pr_ast_features(details, files, repository_path)

            results.append({
                "pr_id":                  pr["id"],
                "pr_number":              pr["number"],
                "title":                  pr["title"],
                "body":                   pr["body"],
                "diff":                   diff_text,
                "additions":              details.get("additions", 0),
                "deletions":              details.get("deletions", 0),
                "files_changed":          details.get("changed_files", 0),
                "commit_count":           details.get("commits", 0),
                "review_duration_hours":  compute_review_time(
                                              details["created_at"],
                                              details["merged_at"]
                                          ),
                "total_comments":         details.get("comments", 0)
                                          + details.get("review_comments", 0),
                "cyclomatic_delta_total": ast_features["cyclomatic_delta_total"],
                "max_nesting_depth":      ast_features["max_nesting_depth"],
                "logic_density_total":    ast_features["logic_density_total"],
                "repo":       repository_path,
                "merged_at": details.get("merged_at", "")
            })
            done_ids.add(pr["id"])

            if len(results) % CHECKPOINT_EVERY == 0:
                check_rate_limit()
                with open(checkpoint_file_path, "w", encoding="utf-8") as f:
                    json.dump(results, f, indent=2, ensure_ascii=False)
                print(f"  Checkpoint saved: {len(results)} PRs")

    print(f"\nRaw PRs seen    : {raw_seen}")
    print(f"Valid collected  : {len(results)}")
    print(f"Skipped          : {skipped}")
    if len(results) < TARGET_VALID_PRS:
        print(f"WARNING: only {len(results)}/{TARGET_VALID_PRS} valid PRs found. "
              f"The repo may not have enough qualifying PRs.")

    with open(out_file_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Dataset written  → {out_file_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository",       required=True,
                        help="owner/name  e.g.  pandas-dev/pandas")
    parser.add_argument("--out_file",         default="pr_dataset.json",
                        help="Output dataset JSON file")
    parser.add_argument("--checkpoint_file",  default="pr_dataset_checkpoint.json",
                        help="Checkpoint file for resuming interrupted runs")
    args = parser.parse_args()
    main(args.repository, args.out_file, args.checkpoint_file)