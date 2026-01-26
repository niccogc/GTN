#!/usr/bin/env python3
# type: ignore
"""
Delete ALL runs and experiments from AIM repository.

WARNING: This is destructive and irreversible!
Use --dry-run first to see what would be deleted.

Usage:
    # Dry run (show what would be deleted)
    python scripts/delete_all_aim_runs.py --aim-repo .aim --dry-run

    # Actually delete from local .aim
    python scripts/delete_all_aim_runs.py --aim-repo .aim

    # Delete from remote AIM server
    python scripts/delete_all_aim_runs.py --aim-repo aim://192.168.5.5:5800

    # Delete from production AIM server
    python scripts/delete_all_aim_runs.py --aim-repo aim://aimtracking.kosmon.org:443
"""

import argparse
import sys


def get_repo(aim_repo: str):
    """Get AIM Repo object for the given repository path/URL."""
    try:
        from aim import Repo
    except ImportError:
        print("ERROR: AIM not installed. Install with: pip install aim")
        sys.exit(1)

    try:
        repo = Repo.from_path(aim_repo)
        return repo
    except Exception as e:
        print(f"ERROR: Failed to connect to AIM repo '{aim_repo}': {e}")
        sys.exit(1)


def count_runs(repo) -> int:
    """Count total runs in the repository."""
    count = 0
    for _ in repo.iter_runs():
        count += 1
    return count


def list_experiments(repo) -> list[str]:
    """List all experiment names in the repository."""
    experiments = set()
    for run in repo.iter_runs():
        try:
            exp = run.experiment
            if exp:
                experiments.add(exp)
        except Exception:
            pass
    return sorted(experiments)


def delete_all_runs(aim_repo: str, dry_run: bool = True) -> tuple[int, list[str]]:
    """
    Delete ALL runs from the AIM repository.

    Returns tuple of (deleted_count, experiment_names).
    """
    repo = get_repo(aim_repo)

    # First, gather info
    print("Scanning repository...")
    experiments = list_experiments(repo)
    total_runs = count_runs(repo)

    print(f"\nRepository: {aim_repo}")
    print(f"Total runs: {total_runs}")
    print(f"Experiments: {len(experiments)}")
    if experiments:
        print("  - " + "\n  - ".join(experiments[:20]))
        if len(experiments) > 20:
            print(f"  ... and {len(experiments) - 20} more")

    if total_runs == 0:
        print("\nNo runs to delete.")
        return 0, experiments

    # Collect all run hashes
    print("\nCollecting run hashes...")
    run_hashes = []
    run_info = []
    for run in repo.iter_runs():
        run_hashes.append(run.hash)
        try:
            exp = run.experiment or "default"
            run_info.append(f"  {run.hash} (experiment: {exp})")
        except Exception:
            run_info.append(f"  {run.hash}")

    # Show what will be deleted
    print(f"\nRuns to delete ({len(run_hashes)}):")
    for info in run_info[:30]:
        print(info)
    if len(run_info) > 30:
        print(f"  ... and {len(run_info) - 30} more")

    if dry_run:
        print("\n" + "=" * 60)
        print("DRY RUN - No runs were deleted")
        print("=" * 60)
        print(f"Would delete {len(run_hashes)} runs")
        print("\nRun without --dry-run to actually delete.")
        return len(run_hashes), experiments

    # Confirm deletion
    print("\n" + "=" * 60)
    print("WARNING: This will permanently delete ALL runs!")
    print("=" * 60)
    confirm = input(f"Type 'DELETE {len(run_hashes)} RUNS' to confirm: ")
    if confirm != f"DELETE {len(run_hashes)} RUNS":
        print("Aborted.")
        return 0, experiments

    # Delete runs
    print(f"\nDeleting {len(run_hashes)} runs...")
    try:
        success, remaining = repo.delete_runs(run_hashes)
        deleted_count = len(run_hashes) - len(remaining)
        print(f"Successfully deleted: {deleted_count}")
        if remaining:
            print(f"Failed to delete: {len(remaining)}")
            for h in remaining[:10]:
                print(f"  - {h}")
        return deleted_count, experiments
    except Exception as e:
        print(f"ERROR during deletion: {e}")
        return 0, experiments


def main():
    parser = argparse.ArgumentParser(
        description="Delete ALL runs and experiments from AIM repository",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Dry run on local .aim directory
    python scripts/delete_all_aim_runs.py --aim-repo .aim --dry-run

    # Delete from local .aim directory
    python scripts/delete_all_aim_runs.py --aim-repo .aim

    # Delete from remote server
    python scripts/delete_all_aim_runs.py --aim-repo aim://aimtracking.kosmon.org:443
        """,
    )
    parser.add_argument(
        "--aim-repo",
        type=str,
        required=True,
        help="AIM repository (local path like '.aim' or URL like 'aim://host:port')",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be deleted without actually deleting",
    )

    args = parser.parse_args()

    if args.dry_run:
        print("=" * 60)
        print("DRY RUN MODE - No runs will be deleted")
        print("=" * 60)

    deleted_count, experiments = delete_all_runs(args.aim_repo, dry_run=args.dry_run)

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    action = "Would delete" if args.dry_run else "Deleted"
    print(f"{action}: {deleted_count} runs")
    print(f"Experiments found: {len(experiments)}")


if __name__ == "__main__":
    main()
