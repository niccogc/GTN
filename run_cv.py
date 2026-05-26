#!/usr/bin/env python
"""
5-Fold Cross-Validation wrapper for GTN/NTN experiments.

Usage:
    python run_cv.py dataset.csv_path=nic.csv model=cpda_typei model.L=4 model.bond_dim=3
"""

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

CSVS_DIR = Path(__file__).parent / "csvs"


def parse_hydra_overrides(args: list[str]) -> dict:
    overrides = {}
    for arg in args:
        if "=" in arg:
            key, value = arg.split("=", 1)
            overrides[key] = value
    return overrides


def run_fold(fold_idx: int, fold_csv: Path, hydra_args: list[str], task: str, sweep_dir: Path, save_model: bool = False) -> dict:
    run_dir = sweep_dir / f"fold_{fold_idx}"
    cmd = [
        sys.executable,
        "run.py",
        "dataset=csv",
        f"dataset.csv_path={fold_csv}",
        f"dataset.task={task}",
        f"hydra.sweep.dir={sweep_dir}",
        f"hydra.run.dir={run_dir}",
        "skip_completed=false",
        f"save_model={str(save_model).lower()}",
    ]
    
    for arg in hydra_args:
        if not arg.startswith("dataset.csv_path="):
            cmd.append(arg)
    
    print(f"\n{'='*60}")
    print(f"Running Fold {fold_idx + 1}/5")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*60}\n")
    
    subprocess.run(cmd, capture_output=False)
    
    results_file = run_dir / "results.json"
    if results_file.exists():
        with open(results_file) as f:
            return json.load(f)
    
    return {"success": False, "fold": fold_idx, "error": "No results file found"}


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--task", type=str, default="regression", choices=["regression", "classification"])
    parser.add_argument("--save-model", action="store_true")
    
    known_args, hydra_args = parser.parse_known_args()
    
    overrides = parse_hydra_overrides(hydra_args)
    csv_path = overrides.get("dataset.csv_path")
    
    if not csv_path:
        print("Error: dataset.csv_path is required")
        print("Usage: python run_cv.py dataset.csv_path=mydata.csv model=cpda_typei ...")
        sys.exit(1)
    
    csv_file = Path(csv_path)
    if not csv_file.exists():
        csv_file = CSVS_DIR / csv_path
    if not csv_file.exists():
        print(f"Error: CSV file not found: {csv_path}")
        sys.exit(1)
    
    print(f"Loading dataset: {csv_file}")
    df = pd.read_csv(csv_file)
    print(f"Dataset shape: {df.shape}")
    
    kf = KFold(n_splits=known_args.n_folds, shuffle=True, random_state=known_args.seed)
    
    target_col = df.columns[-1]
    X = df.drop(columns=[target_col])
    
    overrides = parse_hydra_overrides(hydra_args)
    model_name = overrides.get("model", "unknown")
    csv_name = Path(csv_path).stem
    sweep_dir = Path("outputs") / "kfold" / f"{model_name}_{csv_name}"
    sweep_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Sweep directory: {sweep_dir}")
    
    results = []
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        for fold_idx, (train_val_idx, test_idx) in enumerate(kf.split(X)):
            n_train_val = len(train_val_idx)
            n_val = int(n_train_val * 0.2)
            
            np.random.seed(known_args.seed + fold_idx)
            np.random.shuffle(train_val_idx)
            
            val_idx = train_val_idx[:n_val]
            train_idx = train_val_idx[n_val:]
            
            train_df = df.iloc[train_idx]
            val_df = df.iloc[val_idx]
            test_df = df.iloc[test_idx]
            
            print(f"\nFold {fold_idx + 1}: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")
            
            fold_csv = tmpdir / f"fold_{fold_idx}.csv"
            fold_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
            fold_df.to_csv(fold_csv, index=False)
            
            fold_result = run_fold(fold_idx, fold_csv, hydra_args, known_args.task, sweep_dir, known_args.save_model)
            fold_result["fold"] = fold_idx
            results.append(fold_result)
    
    print("\n" + "="*60)
    print("CROSS-VALIDATION RESULTS")
    print("="*60)
    
    successful = [r for r in results if r.get("success", False)]
    
    if not successful:
        print("No successful folds!")
        for r in results:
            print(f"  Fold {r.get('fold', '?')}: {r.get('error', 'Unknown error')}")
        sys.exit(1)
    
    val_qualities = [r["val_quality"] for r in successful if r.get("val_quality") is not None]
    val_losses = [r["val_loss"] for r in successful if r.get("val_loss") is not None]
    train_qualities = [r["train_quality"] for r in successful if r.get("train_quality") is not None]
    
    print(f"\nSuccessful folds: {len(successful)}/{len(results)}")
    print(f"\nPer-fold validation quality:")
    for r in results:
        status = "✓" if r.get("success") else "✗"
        vq = r.get("val_quality", "N/A")
        vq_str = f"{vq:.4f}" if isinstance(vq, float) else str(vq)
        singular = " (singular)" if r.get("singular") else ""
        print(f"  Fold {r.get('fold', '?')}: {status} val_quality={vq_str}{singular}")
    
    if val_qualities:
        print(f"\nAggregated Results:")
        print(f"  Val Quality:   {np.mean(val_qualities):.4f} ± {np.std(val_qualities):.4f}")
        print(f"  Val Loss:      {np.mean(val_losses):.4f} ± {np.std(val_losses):.4f}")
        print(f"  Train Quality: {np.mean(train_qualities):.4f} ± {np.std(train_qualities):.4f}")
    
    output_file = sweep_dir / "cv_results.json"
    
    summary = {
        "n_folds": known_args.n_folds,
        "successful_folds": len(successful),
        "val_quality_mean": float(np.mean(val_qualities)) if val_qualities else None,
        "val_quality_std": float(np.std(val_qualities)) if val_qualities else None,
        "val_loss_mean": float(np.mean(val_losses)) if val_losses else None,
        "val_loss_std": float(np.std(val_losses)) if val_losses else None,
        "train_quality_mean": float(np.mean(train_qualities)) if train_qualities else None,
        "train_quality_std": float(np.std(train_qualities)) if train_qualities else None,
        "per_fold_results": results,
        "hydra_args": hydra_args,
    }
    
    with open(output_file, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    
    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()
