#!/usr/bin/env python3
# type: ignore
from aim import Repo
from tqdm import tqdm

repo = Repo.from_path("aim://192.168.5.5:5800")

run_hashes_to_delete = []

for run in tqdm(repo.iter_runs()):
    try:
        summary = run.get("summary")
        if summary and isinstance(summary, dict):
            test_loss = summary.get("test_loss", "NOT_SET")
            if test_loss is None:
                print(f"Found: {run.hash}")
                run_hashes_to_delete.append(run.hash)
    except Exception as e:
        print(f"Error on {run.hash}: {e}")
        continue

print(f"\nTotal runs with None test_loss: {len(run_hashes_to_delete)}")

if run_hashes_to_delete:
    success, remaining = repo.delete_runs(run_hashes_to_delete)
    print(f"Deleted {len(run_hashes_to_delete) - len(remaining)} runs")
else:
    print("No runs to delete")
