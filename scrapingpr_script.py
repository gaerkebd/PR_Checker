import os
import json
from github import Github

# ===== CONFIG =====
GITHUB_TOKEN = "your_token_here"
REPO_NAME = "facebook/react"   # change this
MAX_PRS = 100                  # keep small for now

# ==================

g = Github(GITHUB_TOKEN)
repo = g.get_repo(REPO_NAME)

data = []

pulls = repo.get_pulls(state="closed")

for i, pr in enumerate(pulls):
    if i >= MAX_PRS:
        break

    print(f"Processing PR #{pr.number}")

    pr_dict = {
        "number": pr.number,
        "title": pr.title,
        "body": pr.body,
        "diff_url": pr.diff_url,
        "comments": [],
        "reviews": []
    }

    # Comments
    for c in pr.get_comments():
        pr_dict["comments"].append({
            "user": c.user.login,
            "body": c.body
        })

    # Reviews
    for r in pr.get_reviews():
        pr_dict["reviews"].append({
            "user": r.user.login,
            "state": r.state,
            "body": r.body
        })

    data.append(pr_dict)

# Save
with open("pr_data.json", "w") as f:
    json.dump(data, f, indent=2)

print("Done.")
