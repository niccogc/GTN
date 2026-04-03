# Cluster Workflow

This document explains how to run experiments across multiple clusters (titans, hpc) while managing file quotas.

## Overview

- **main**: Source of truth. Contains all outputs and the tracking CSV.
- **titans** / **hpc**: Cluster branches. Run experiments, push outputs to main, then delete locally to free quota.

The `runs_tracking.csv` tracks all completed runs. It's regenerated on main from outputs using `scripts/backfill_tracking.py`. Clusters read it to skip already-completed runs.

## Initial Setup (on each cluster)

```bash
# Clone or fetch latest
git fetch origin

# Switch to your cluster branch
git checkout titans   # or: git checkout hpc

# Delete local main branch (not needed on cluster)
git branch -d main

# Set upstream tracking
git branch --set-upstream-to=origin/titans   # or: origin/hpc
```

## Running Experiments

```bash
# 1. Get the latest tracking CSV from main
git fetch origin main
git checkout origin/main -- runs_tracking.csv

# 2. Run experiments (skips runs already in CSV)
python run.py --multirun ...

# Runs are tracked via:
#   - runs_tracking.csv (from main, read-only on clusters)
#   - local results.json files (fallback for recent local runs)
```

## Pushing Results to Main

Use the provided script:

```bash
# Push results and clean up (all in one)
./scripts/push_results.sh "Description of experiments"
```

The script does:
1. `git add .` (stages outputs + logs)
2. `git commit`
3. `git push origin <branch>`
4. Asks confirmation, then:
   - `rm -rf outputs/` (frees quota)
   - `git checkout -- outputs/` (restores git's view so deletions aren't staged)

Merges from cluster branches to main are done locally on your PC.

## Getting Updates from Main

**Never merge main** - it would pull all outputs and exceed quota.

Use the sync script to pull all code changes (excluding outputs):
```bash
./scripts/sync_from_main.sh
```

## Regenerating the Tracking CSV (Main Only)

On the main machine (which has all outputs):

```bash
git checkout main
python scripts/backfill_tracking.py
git add runs_tracking.csv
git commit -m "Update tracking CSV"
git push origin main
```

## Summary

| Action | Command |
|--------|---------|
| Sync from main | `./scripts/sync_from_main.sh` |
| Run experiments | `python run.py --multirun ...` |
| Push + cleanup | `./scripts/push_results.sh "msg"` |

## Config

In `conf/config.yaml`:
- `skip_completed: true` - Skip runs found in CSV or results.json
- `update_tracking: false` - Clusters don't modify CSV (set `true` on main only)
