#!/usr/bin/env python3
# type: ignore
import json
import os
from pathlib import Path
from tqdm import tqdm

results_dir = Path("results")

deleted_count = 0
total_count = 0

all_json_files = list(results_dir.rglob("*.json"))

for json_file in tqdm(all_json_files):
    if json_file.name == "summary.json":
        continue

    total_count += 1

    try:
        with open(json_file) as f:
            data = json.load(f)

        test_loss = data.get("test_loss")
        if test_loss is None:
            print(f"Deleting: {json_file}")
            os.remove(json_file)
            deleted_count += 1
    except Exception as e:
        print(f"Error on {json_file}: {e}")
        continue

print(f"\nTotal files checked: {total_count}")
print(f"Deleted: {deleted_count}")
