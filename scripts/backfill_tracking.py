#!/usr/bin/env python3
"""Backfill runs_tracking.csv from existing outputs directory.

This script scans the outputs directory for results.json files and
populates the tracking CSV with all completed runs.

Usage:
    python scripts/backfill_tracking.py
    python scripts/backfill_tracking.py --outputs-dir outputs --dry-run
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
from datetime import datetime
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

# Must match utils/tracking.py
TRACKING_COLUMNS = [
    "run_id",
    "trainer_type",
    "dataset",
    "model",
    "L",
    "bond_dim",
    "ridge",
    "init_strength",
    "reduction_factor",
    "seed",
    "success",
    "singular",
    "oom_error",
    "val_quality",
    "train_quality",
    "best_epoch",
    "output_path",
    "timestamp",
]


def extract_run_info(results_path: Path) -> dict | None:
    """Extract run information from a results.json file.
    
    Args:
        results_path: Path to results.json file
        
    Returns:
        Dict with tracking info, or None if extraction failed
    """
    try:
        with open(results_path) as f:
            data = json.load(f)
    except (json.JSONDecodeError, FileNotFoundError) as e:
        log.warning(f"  Could not read {results_path}: {e}")
        return None
    
    # Extract config
    config = data.get("config", {})
    if not config:
        log.warning(f"  No config found in {results_path}")
        return None
    
    # Extract parameters from config
    trainer_cfg = config.get("trainer", {})
    model_cfg = config.get("model", {})
    dataset_cfg = config.get("dataset", {})
    
    trainer_type = trainer_cfg.get("type")
    dataset = dataset_cfg.get("name")
    model = model_cfg.get("name")
    L = model_cfg.get("L")
    bond_dim = model_cfg.get("bond_dim")
    ridge = trainer_cfg.get("ridge")
    init_strength = model_cfg.get("init_strength")
    seed = config.get("seed")
    reduction_factor = model_cfg.get("reduction_factor") if "LMPO" in model else 1 
    
    # Validate required fields
    if not all([trainer_type, dataset, model, L is not None, bond_dim is not None, 
                ridge is not None, init_strength is not None, seed is not None]):
        log.warning(f"  Missing required config fields in {results_path}")
        log.warning(f"    trainer_type={trainer_type}, dataset={dataset}, model={model}")
        log.warning(f"    L={L}, bond_dim={bond_dim}, ridge={ridge}, init_strength={init_strength}, seed={seed}")
        return None
    
    # Generate run_id
    run_id = (
        f"{trainer_type}_{dataset}_{model}"
        f"_L{L}_bd{bond_dim}"
        f"_rg{ridge}_init{init_strength}"
        f"_s{seed}"
        f"_rf{reduction_factor}"
    )
    
    # Get output path relative to cwd
    output_path = results_path.parent
    try:
        output_path = output_path.relative_to(Path.cwd())
    except ValueError:
        pass  # Keep absolute if not relative to cwd
    
    try:
        mtime = results_path.stat().st_mtime
        timestamp = datetime.fromtimestamp(mtime).isoformat()
    except OSError:
        timestamp = datetime.now().isoformat()
    
    return {
        "run_id": run_id,
        "trainer_type": trainer_type,
        "dataset": dataset,
        "model": model,
        "L": L,
        "bond_dim": bond_dim,
        "ridge": ridge,
        "init_strength": init_strength,
        "seed": seed,
        "success": data.get("success", False),
        "singular": data.get("singular", False),
        "oom_error": data.get("oom_error", False),
        "val_quality": data.get("val_quality"),
        "train_quality": data.get("train_quality"),
        "best_epoch": data.get("best_epoch"),
        "output_path": str(output_path),
        "timestamp": timestamp,
    }


def find_results_files(outputs_dir: Path) -> list[Path]:
    """Find all results.json files in the outputs directory."""
    return list(outputs_dir.rglob("results.json"))


def main():
    parser = argparse.ArgumentParser(description="Backfill tracking CSV from existing outputs")
    parser.add_argument(
        "--outputs-dir",
        type=Path,
        default=Path("outputs"),
        help="Path to outputs directory (default: outputs)",
    )
    parser.add_argument(
        "--tracking-file",
        type=Path,
        default=Path("runs_tracking.csv"),
        help="Path to tracking CSV file (default: runs_tracking.csv)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be done without writing",
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help="Append to existing tracking file instead of overwriting",
    )
    parser.add_argument(
        "--verbose",
        help="NEVER DO IT",
    )
    args = parser.parse_args()
    
    if not args.outputs_dir.exists():
        log.error(f"Outputs directory not found: {args.outputs_dir}")
        return 1
    
    # Find all results.json files
    log.info(f"Scanning {args.outputs_dir} for results.json files...")
    results_files = find_results_files(args.outputs_dir)
    log.info(f"Found {len(results_files)} results.json files")
    
    # Extract run info from each
    runs = []
    seen_run_ids = set()
    
    for results_path in results_files:
        if args.verbose:
            log.info(f"Processing: {results_path}")
        run_info = extract_run_info(results_path)
        
        if run_info is None:
            continue
            
        run_id = run_info["run_id"]
        if run_id in seen_run_ids:
            log.warning(f"  Duplicate run_id: {run_id}, keeping first occurrence")
            continue
            
        seen_run_ids.add(run_id)
        runs.append(run_info)
        if args.verbose:
            log.info(f"  -> {run_id} (success={run_info['success']}, val_quality={run_info['val_quality']})")
    log.info(f"\nExtracted {len(runs)} unique runs")
    
    # Summary stats
    successful = sum(1 for r in runs if r["success"])
    singular = sum(1 for r in runs if r["singular"])
    oom = sum(1 for r in runs if r["oom_error"])
    log.info(f"  Successful: {successful}")
    log.info(f"  Singular:   {singular}")
    log.info(f"  OOM:        {oom}")
    
    if args.dry_run:
        log.info("\n[DRY RUN] Would write to tracking file, but --dry-run specified")
        return 0
    
    # Write to tracking file
    mode = "a" if args.append else "w"
    file_exists = args.tracking_file.exists() and args.append
    
    with open(args.tracking_file, mode, newline="") as f:
        writer = csv.DictWriter(f, fieldnames=TRACKING_COLUMNS)
        if not file_exists:
            writer.writeheader()
        writer.writerows(runs)
    
    log.info(f"\nWrote {len(runs)} runs to {args.tracking_file}")
    return 0


if __name__ == "__main__":
    exit(main())
