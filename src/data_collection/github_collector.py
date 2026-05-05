"""
github_collector.py
--------------------
Fetches PR data (diff, comments, review comments) from a GitHub repository
using the PyGithub library. Handles pagination and rate limits automatically.
"""

import json
import os
import time
import requests
from typing import Optional

from github import Github, RateLimitExceededException
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type

from src.utils import load_config, get_logger

logger = get_logger(__name__)

def _fetch_diff(diff_url: str, token: str) -> str:
    """Download the raw unified diff for a PR."""
    headers = {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github.v3.diff",
    }
    resp = requests.get(diff_url, headers=headers, timeout=30)
    resp.raise_for_status()
    return resp.text


@retry(
    retry=retry_if_exception_type(RateLimitExceededException),
    wait=wait_exponential(multiplier=60, min=60, max=600),
    stop=stop_after_attempt(5),
)
def _safe_get_review_comments(pr):
    """Wrap PR.get_review_comments() with retry on rate-limit."""
    return list(pr.get_review_comments())


@retry(
    retry=retry_if_exception_type(RateLimitExceededException),
    wait=wait_exponential(multiplier=60, min=60, max=600),
    stop=stop_after_attempt(5),
)
def _safe_get_reviews(pr):
    """Wrap PR.get_reviews() with retry on rate-limit."""
    return list(pr.get_reviews())


def collect_prs(config: dict, github_token: str) -> list[dict]:
    """
    Main collection routine.

    Parameters
    ----------
    config       : loaded config dict (from config/config.yaml)
    github_token : personal access token with repo read scope

    Returns
    -------
    list of raw PR dicts
    """
    gh_cfg = config["github"]
    repo_name = gh_cfg["repo"]
    max_prs = gh_cfg["max_prs"]
    print("collecting PRs from " + repo_name)
    min_comments = gh_cfg["min_review_comments"]
    state = gh_cfg.get("state", "closed")

    g = Github(github_token)
    repo = g.get_repo(repo_name)
    logger.info(f"Fetching up to {max_prs} {state} PRs from {repo_name}")

    results = []
    pulls = repo.get_pulls(state=state, sort="updated", direction="desc")
    count = 0
    
    for pr in pulls:

        if len(results) >= max_prs:
            break

        # Check rate limit proactively
        remaining, _ = g.rate_limiting
        if remaining < 20:
            reset_ts = g.rate_limiting_resettime  # Unix timestamp
            sleep_s = max(0, reset_ts - time.time()) + 5
            logger.warning(f"Rate limit low ({remaining}). Sleeping {sleep_s:.0f}s …")
            time.sleep(sleep_s)

        try:
            review_comments = _safe_get_review_comments(pr)
        except Exception as e:
            logger.warning(f"PR #{pr.number}: failed to fetch review comments — {e}")
            continue

        if len(review_comments) < min_comments:
            logger.debug(f"PR #{pr.number}: skipped (only {len(review_comments)} review comments)")
            continue

        logger.info(f"{count}/{max_prs} | Collecting PR #{pr.number}: {pr.title!r}")

        try:
            diff_text = _fetch_diff(pr.diff_url, github_token)
        except Exception as e:
            logger.warning(f"PR #{pr.number}: could not fetch diff — {e}")
            diff_text = ""

        try:
            reviews = _safe_get_reviews(pr)
        except Exception as e:
            logger.warning(f"PR #{pr.number}: could not fetch reviews — {e}")
            reviews = []

        pr_record = {
            "pr_number": pr.number,
            "repo": repo_name,
            "title": pr.title,
            "body": pr.body or "",
            "diff": diff_text,
            "state": pr.state,
            "created_at": pr.created_at.isoformat(),
            "merged": pr.merged,
            "review_comments": [
                {
                    "user": rc.user.login if rc.user else "unknown",
                    "body": rc.body,
                    "path": rc.path,
                    "line": rc.original_line,
                    "diff_hunk": rc.diff_hunk,
                }
                for rc in review_comments
            ],
            "reviews": [
                {
                    "user": r.user.login if r.user else "unknown",
                    "state": r.state,
                    "body": r.body or "",
                }
                for r in reviews
            ],
        }
        count += 1
        results.append(pr_record)
        

    logger.info(f"Collected {len(results)} PRs.")
    return results


def save_raw(data: list[dict], path: str) -> None:
    """Persist raw PR data to disk as JSON."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    logger.info(f"Raw data saved → {path}")
