#!/usr/bin/env python3
# type: ignore
"""
Delete runs with errors OTHER THAN singular matrix errors.

This script cleans up:
1. JSON result files in results/ directory that have errors (but are NOT singular matrix errors)
2. Corresponding AIM runs with the same errors

Runs with singular=True are KEPT - those are valid completed runs.
Only runs with 'error' field that is NOT a singular matrix error are deleted.
"""

import argparse
import json
import os
from pathlib import Path
from tqdm import tqdm

SINGULAR_KEYWORDS = ["singular", "linalg", "cholesky", "positive definite", "not invertible"]


def is_singular_error(error_msg: str) -> bool:
    """Check if an error message indicates a singular matrix error."""
    if error_msg is None:
        return False
    error_lower = error_msg.lower()
    return any(keyword in error_lower for keyword in SINGULAR_KEYWORDS)


def delete_json_results(results_dir: str, dry_run: bool = True) -> list[str]:
    """
    Delete JSON result files that have non-singular errors.

    Returns list of deleted file paths (or would-be-deleted if dry_run).
    """
    results_path = Path(results_dir)
    deleted_files = []

    if not results_path.exists():
        print(f"Results directory not found: {results_dir}")
        return deleted_files

    all_json_files = list(results_path.rglob("*.json"))

    for json_file in tqdm(all_json_files, desc="Scanning JSON files"):
        if json_file.name == "summary.json":
            continue

        try:
            with open(json_file) as f:
                data = json.load(f)

            error = data.get("error")
            singular = data.get("singular", False)

            if error is not None and not singular and not is_singular_error(error):
                deleted_files.append(str(json_file))
                if not dry_run:
                    print(f"Deleting: {json_file}")
                    print(f"  Error: {error[:100]}...")
                    os.remove(json_file)
                else:
                    print(f"[DRY-RUN] Would delete: {json_file}")
                    print(f"  Error: {error[:100]}...")

        except Exception as e:
            print(f"Error reading {json_file}: {e}")
            continue

    return deleted_files


def delete_aim_runs(aim_repo: str, dry_run: bool = True) -> list[str]:
    """
    Delete AIM runs that have non-singular errors.

    Returns list of deleted run hashes (or would-be-deleted if dry_run).
    """
    try:
        from aim import Repo
    except ImportError:
        print("AIM not installed. Skipping AIM cleanup.")
        return []

    try:
        repo = Repo.from_path(aim_repo)
    except Exception as e:
        print(f"Failed to connect to AIM repo '{aim_repo}': {e}")
        return []

    run_hashes_to_delete = []

    for run in tqdm(repo.iter_runs(), desc="Scanning AIM runs"):
        try:
            summary = run.get("summary")
            if summary and isinstance(summary, dict):
                error = summary.get("error")
                singular = summary.get("singular", False)

                if error is not None and not singular and not is_singular_error(error):
                    run_hashes_to_delete.append(run.hash)
                    if dry_run:
                        print(f"[DRY-RUN] Would delete AIM run: {run.hash}")
                        print(
                            f"  Error: {error[:100] if isinstance(error, str) else str(error)[:100]}..."
                        )
                    else:
                        print(f"Will delete AIM run: {run.hash}")
                        print(
                            f"  Error: {error[:100] if isinstance(error, str) else str(error)[:100]}..."
                        )

        except Exception as e:
            print(f"Error checking run {run.hash}: {e}")
            continue

    if run_hashes_to_delete and not dry_run:
        print(f"\nDeleting {len(run_hashes_to_delete)} AIM runs...")
        success, remaining = repo.delete_runs(run_hashes_to_delete)
        deleted_count = len(run_hashes_to_delete) - len(remaining)
        print(f"Deleted {deleted_count} runs, {len(remaining)} failed")

    return run_hashes_to_delete


def main():
    parser = argparse.ArgumentParser(
        description="Delete runs with errors OTHER THAN singular matrix errors"
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="results",
        help="Results directory to clean (default: results)",
    )
    parser.add_argument(
        "--aim-repo",
        type=str,
        default=None,
        help="AIM repository URL (e.g., aim://192.168.5.5:5800)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be deleted without actually deleting",
    )
    parser.add_argument(
        "--skip-json",
        action="store_true",
        help="Skip JSON file deletion",
    )
    parser.add_argument(
        "--skip-aim",
        action="store_true",
        help="Skip AIM run deletion",
    )

    args = parser.parse_args()

    if args.dry_run:
        print("=" * 60)
        print("DRY RUN MODE - No files will be deleted")
        print("=" * 60)

    deleted_json = []
    deleted_aim = []

    if not args.skip_json:
        print("\n--- Scanning JSON result files ---")
        deleted_json = delete_json_results(args.results_dir, dry_run=args.dry_run)

    if not args.skip_aim and args.aim_repo:
        print("\n--- Scanning AIM runs ---")
        deleted_aim = delete_aim_runs(args.aim_repo, dry_run=args.dry_run)

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    action = "Would delete" if args.dry_run else "Deleted"
    print(f"{action} JSON files: {len(deleted_json)}")
    print(f"{action} AIM runs: {len(deleted_aim)}")

    if args.dry_run:
        print("\nRun without --dry-run to actually delete these files.")


if __name__ == "__main__":
    main()
