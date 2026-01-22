#!/usr/bin/env python3
# type: ignore
"""
Delete aim runs with invalid/missing metrics.

Deletes runs where:
- test_loss is None
- test_quality is "-" or not a valid number
"""

import math
from aim import Repo
from tqdm import tqdm


def is_invalid_number(value):
    """Check if value is invalid (None, "-", NaN, or not a number)."""
    if value is None:
        return True
    if isinstance(value, str):
        value = value.strip()
        if value == "-" or value == "":
            return True
        try:
            float(value)
            return False
        except (ValueError, TypeError):
            return True
    if isinstance(value, float):
        return math.isnan(value)
    if isinstance(value, (int, float)):
        return False
    return True


repo = Repo.from_path("aim://192.168.5.5:5800")

run_hashes_to_delete = []

for run in tqdm(repo.iter_runs()):
    try:
        summary = run.get("summary")
        if summary and isinstance(summary, dict):
            test_loss = summary.get("test_loss", "NOT_SET")
            test_quality = summary.get("test_quality", "NOT_SET")

            should_delete = False
            reason = []

            if test_loss is None:
                should_delete = True
                reason.append("test_loss=None")

            if test_quality != "NOT_SET" and is_invalid_number(test_quality):
                should_delete = True
                reason.append(f"test_quality={test_quality!r}")

            if should_delete:
                print(f"Found: {run.hash} ({', '.join(reason)})")
                run_hashes_to_delete.append(run.hash)
    except Exception as e:
        print(f"Error on {run.hash}: {e}")
        continue

print(f"\nTotal runs to delete: {len(run_hashes_to_delete)}")

if run_hashes_to_delete:
    success, remaining = repo.delete_runs(run_hashes_to_delete)
    print(f"Deleted {len(run_hashes_to_delete) - len(remaining)} runs")
else:
    print("No runs to delete")
