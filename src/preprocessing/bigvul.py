"""
bigvul.py
---------
Loads the BigVul vulnerability dataset from HuggingFace and transforms each
row into a standardized record that can be used for training or fine-tuning
a vulnerability-detection model.


Usage
-----
    from src.preprocessing.bigvul import load_from_huggingface, save_records

    records = load_from_huggingface(max_records=5000)
    save_records(records, "data/raw/bigvul/bigvul_records.json")
"""

import json
import os
from typing import TypedDict
from datasets import load_dataset

# HuggingFace dataset identifier for BigVul
dataset = load_dataset("bstee615/bigvul", split="train", streaming=True)
bigvul_records = []

for i, row in enumerate(dataset):
    bigvul_records.append({
        "cve_id": row["CVE ID"],
        "cve_reference": row["CVE Page"],
        "cwe_id": row["CWE ID"],
        "github_commit_link": row["codeLink"],
        "commit_message": row["commit_message"],
        "func_after": row["func_after"],
        "func_before": row["func_before"],
        "lang": row["lang"],
        "vuln_num": row["vul"]
    })
    print(f"Processed record {i+1}: {bigvul_records[-1]['cve_id']}")
    if i >= 5000:  # Limit to 5000 records for now
        break
    


