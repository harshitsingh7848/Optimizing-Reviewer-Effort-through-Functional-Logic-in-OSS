import requests
from config import config
import datetime
import json
import base64
import time
from get_cyclomatic_complexity import cyclomatic_complexity_total
from get_nesting_depth import max_nesting_depth
from get_logic_density import logic_density

HEADERS = {
    "Accept": "application/vnd.github+json",
    "Authorization": f'Bearer {config["API_TOKEN"]}',
    "X-GitHub-Api-Version": "2022-11-28",
}

def get_pull_requests(max_raw: int = 1500):
    url = f'{config["BASE_URL"]}/repos/pandas-dev/pandas/pulls'
    all_prs = []
    page = 1

    while len(all_prs) < max_raw:
        params = {
            "state": "closed",
            "per_page": 100,
            "page": page
        }
        try:
            response = requests.get(url, headers=HEADERS, params=params)
            response.raise_for_status()
            batch = response.json()
            if not batch:
                break
            all_prs.extend(batch)
            print(f"Fetched {len(all_prs)} raw PRs so far...")
            page += 1
            time.sleep(0.5)
        except Exception as error:
            print("error: ", error)
            break

    return all_prs


def get_pr_details(pr_number, retries=3):
    url = f'{config["BASE_URL"]}/repos/pandas-dev/pandas/pulls/{pr_number}'
    for attempt in range(retries):
        try:
            response = requests.get(url, headers=HEADERS, timeout=30)
            response.raise_for_status()
            return response.json()
        except Exception as error:
            print(f"Attempt {attempt+1}/{retries} failed for PR details {pr_number}: {error}")
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
            else:
                return {}

def get_pr_files(pr_number, retries=3):
    url = f'{config["BASE_URL"]}/repos/pandas-dev/pandas/pulls/{pr_number}/files'
    for attempt in range(retries):
        try:
            response = requests.get(url, headers=HEADERS, timeout=30)
            response.raise_for_status()
            return response.json()
        except Exception as error:
            print(f"Attempt {attempt+1}/{retries} failed for PR files {pr_number}: {error}")
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
            else:
                return []

def get_pr_comments(pr_number):
    url = f'{config["BASE_URL"]}/repos/pandas-dev/pandas/issues/{pr_number}/comments'
    response = requests.get(url, headers=HEADERS)
    return response.json()


def compute_review_time(created, merged):
    if not merged:
        return 0
    t1 = datetime.datetime.strptime(created, "%Y-%m-%dT%H:%M:%SZ")
    t2 = datetime.datetime.strptime(merged, "%Y-%m-%dT%H:%M:%SZ")
    return (t2 - t1).total_seconds() / 3600

def get_file_content_at_ref(file_path, ref, retries=3):
    url = f'{config["BASE_URL"]}/repos/pandas-dev/pandas/contents/{file_path}'
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

            content = data.get("content", "")
            encoding = data.get("encoding", "")

            if encoding == "base64":
                return base64.b64decode(content).decode("utf-8", errors="ignore")

            return None

        except Exception as error:
            print(f"Attempt {attempt+1}/{retries} failed for {file_path}: {error}")
            if attempt < retries - 1:
                time.sleep(2 ** attempt)  # wait 1s, 2s, 4s between retries
            else:
                return None  # give up after 3 attempts, return None gracefully

def compute_file_features(code):
    if not code or not code.strip():
        return {
            "cyclomatic": 0,
            "nesting": 0,
            "logic_density": 0.0,
        }

    cyclomatic = cyclomatic_complexity_total(code)
    nesting = max_nesting_depth(code)
    density = logic_density(code)

    return {
        "cyclomatic": 0 if cyclomatic is None else cyclomatic,
        "nesting": 0 if nesting is None else nesting,
        "logic_density": 0.0 if density is None else density,
    }

def compute_pr_ast_features(details, files):
    base_sha = details["base"]["sha"]
    head_sha = details["head"]["sha"]

    cyclomatic_delta_total = 0
    max_nesting_depth_total = 0
    logic_density_values = []

    for f in files:
        file_path = f.get("filename", "")

        if not file_path.endswith(".py"):
            continue

        before_code = get_file_content_at_ref(file_path, base_sha)
        after_code = get_file_content_at_ref(file_path, head_sha)

        before_features = compute_file_features(before_code or "")
        after_features = compute_file_features(after_code or "")

        cyclomatic_delta_total += (
            after_features["cyclomatic"] - before_features["cyclomatic"]
        )

        max_nesting_depth_total = max(
            max_nesting_depth_total,
            after_features["nesting"]
        )

        logic_density_values.append(after_features["logic_density"])

    logic_density_total = (
        sum(logic_density_values) / len(logic_density_values)
        if logic_density_values else 0.0
    )

    return {
        "cyclomatic_delta_total": cyclomatic_delta_total,
        "max_nesting_depth": max_nesting_depth_total,
        "logic_density_total": logic_density_total,
    }

BOT_PATTERNS = [
    "bot", "dependabot", "github-actions", 
    "renovate", "allcontributors", "codecov"
]

def is_bot_pr(pr: dict) -> bool:
    login = (pr.get("user") or {}).get("login", "").lower()
    return any(pattern in login for pattern in BOT_PATTERNS)

def is_doc_only_pr(files: list) -> bool:
    """Returns True if PR only modifies non-code files"""
    code_extensions = {".py", ".c", ".cpp", ".java", ".js", ".ts"}
    for f in files:
        filename = f.get("filename", "")
        if any(filename.endswith(ext) for ext in code_extensions):
            return False
    return True  # no code files found

def is_dependency_only_pr(pr: dict) -> bool:
    title = pr.get("title", "").lower()
    dependency_keywords = ["bump", "update dependency", "upgrade", "dependabot"]
    return any(kw in title for kw in dependency_keywords)

def is_empty_diff_pr(files: list) -> bool:
    if not files:
        return True
    return all(f.get("changes", 0) == 0 for f in files)


def check_rate_limit():
    url = f'{config["BASE_URL"]}/rate_limit'
    resp = requests.get(url, headers=HEADERS).json()
    remaining = resp["rate"]["remaining"]
    reset_time = resp["rate"]["reset"]
    if remaining < 100:
        wait = reset_time - time.time() + 10
        print(f"Rate limit low ({remaining} remaining). Waiting {wait:.0f}s...")
        time.sleep(max(wait, 0))
    else:
        print(f"Rate limit: {remaining} requests remaining")

def main():
    results = []
    try:
        with open("pr_dataset_checkpoint.json", encoding="utf-8") as f:
            results = json.load(f)
        done_ids = {r["pr_id"] for r in results}
        print(f"Resuming from checkpoint: {len(results)} PRs already collected")
    except FileNotFoundError:
        results = []
        done_ids = set()
    
    skipped = 0
    pull_requests = get_pull_requests(max_raw=1500)

    for pr in pull_requests:
        if len(results) >= 500:
            break
        pr_number = pr["number"]
        print(f'{pr_number=}')

        if pr["id"] in done_ids:
            continue

        if pr.get("draft", False):
            skipped += 1
            continue
        if is_bot_pr(pr):
            skipped += 1
            continue
        if is_dependency_only_pr(pr):
            skipped += 1
            continue
        

        details = get_pr_details(pr_number)
        files = get_pr_files(pr_number)

        if is_empty_diff_pr(files):
            skipped += 1
            continue
        if is_doc_only_pr(files):
            skipped += 1
            continue
        if not details.get("merged_at"):
            skipped += 1
            continue
        print(f"Valid PR {len(results)+1}/500: {pr_number}")
        # comments = get_pr_comments(pr_number)

        row = {}

        row["pr_id"] = pr["id"]
        row["title"] = pr["title"]
        row["body"] = pr["body"]

        diff_text = ""
        for f in files:
            if "patch" in f:
                diff_text += f["patch"] + "\n"

        row["diff"] = diff_text

        row["additions"] = details.get("additions", 0)
        row["deletions"] = details.get("deletions", 0)
        row["files_changed"] = details.get("changed_files", 0)
        row["commit_count"] = details.get("commits", 0)

        row["review_duration_hours"] = compute_review_time(
            details["created_at"],
            details["merged_at"]
        )

        row["total_comments"] = details.get("comments", 0) + details.get("review_comments", 0)

        ast_features = compute_pr_ast_features(details, files)

        row["cyclomatic_delta_total"] = ast_features["cyclomatic_delta_total"]
        row["max_nesting_depth"] = ast_features["max_nesting_depth"]
        row["logic_density_total"] = ast_features["logic_density_total"]

        results.append(row)
        time.sleep(0.5)
        if len(results) % 100 == 0:
            check_rate_limit()
            # also save checkpoint
            with open("pr_dataset_checkpoint.json", "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            print(f"Checkpoint saved at {len(results)} PRs")
    print(f"\nTotal valid PRs collected: {len(results)}")
    print(f"Total skipped: {skipped}")


    if len(results) < 500:
        print(f"WARNING: Only collected {len(results)} valid PRs.")
        print(f"Increase max_raw (currently 1500) and rerun.")

    with open("pr_dataset.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    main()
