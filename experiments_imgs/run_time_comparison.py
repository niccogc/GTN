#!/usr/bin/env python
"""
Run NTN vs GTN time comparison on MNIST with CMPO2.

Runs both trainers, collects metrics, and plots val_accuracy vs wall_time.
"""
import json
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt


def run_experiment(trainer: str, output_base: Path) -> dict:
    output_dir = output_base / trainer
    output_dir.mkdir(parents=True, exist_ok=True)
    
    cmd = [
        sys.executable, "run.py",
        "+experiment=time_comparison",
        f"trainer={trainer}",
        f"hydra.run.dir={output_dir}",
    ]
    
    print(f"\n{'='*60}")
    print(f"Running {trainer.upper()}...")
    print(f"{'='*60}\n")
    
    result = subprocess.run(cmd, cwd=Path(__file__).parent)
    
    if result.returncode != 0:
        print(f"Warning: {trainer} run failed with code {result.returncode}")
        return None
    
    results_file = output_dir / "results.json"
    if not results_file.exists():
        print(f"Warning: No results file found at {results_file}")
        return None
    
    with open(results_file) as f:
        return json.load(f)


def plot_comparison(ntn_result: dict, gtn_result: dict, output_path: Path):
    fig, ax = plt.subplots(figsize=(10, 6))
    
    if ntn_result and ntn_result.get("metrics_log"):
        ntn_metrics = ntn_result["metrics_log"]
        ntn_times = [m["wall_time"] for m in ntn_metrics]
        ntn_acc = [m["val_accuracy"] for m in ntn_metrics]
        ax.plot(ntn_times, ntn_acc, 'o-', label=f'NTN (ALS)', markersize=4, linewidth=1.5)
    
    if gtn_result and gtn_result.get("metrics_log"):
        gtn_metrics = gtn_result["metrics_log"]
        gtn_times = [m["wall_time"] for m in gtn_metrics]
        gtn_acc = [m["val_accuracy"] for m in gtn_metrics]
        ax.plot(gtn_times, gtn_acc, 's-', label=f'GTN (Gradient)', markersize=3, linewidth=1.5, alpha=0.8)
    
    ax.set_xlabel("Wall Time (seconds)", fontsize=12)
    ax.set_ylabel("Validation Accuracy", fontsize=12)
    ax.set_title("NTN vs GTN: Validation Accuracy over Time\n(MNIST, CMPO2, bond_dim=10)", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"\nPlot saved to: {output_path}")
    plt.show()


def main():
    output_base = Path(__file__).parent / "outputs" / "time_comparison"
    output_base.mkdir(parents=True, exist_ok=True)
    
    ntn_result = run_experiment("ntn", output_base)
    gtn_result = run_experiment("gtn", output_base)
    
    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)
    
    if ntn_result:
        print(f"\nNTN:")
        print(f"  Val Accuracy: {ntn_result.get('val_accuracy', 'N/A'):.4f}")
        print(f"  Test Accuracy: {ntn_result.get('test_accuracy', 'N/A'):.4f}")
        print(f"  Total Time: {ntn_result.get('total_time', 'N/A'):.2f}s")
        print(f"  Best Epoch: {ntn_result.get('best_epoch', 'N/A')}")
    
    if gtn_result:
        print(f"\nGTN:")
        print(f"  Val Accuracy: {gtn_result.get('val_accuracy', 'N/A'):.4f}")
        print(f"  Test Accuracy: {gtn_result.get('test_accuracy', 'N/A'):.4f}")
        print(f"  Total Time: {gtn_result.get('total_time', 'N/A'):.2f}s")
        print(f"  Best Epoch: {gtn_result.get('best_epoch', 'N/A')}")
    
    plot_path = output_base / "time_comparison.png"
    plot_comparison(ntn_result, gtn_result, plot_path)
    
    combined_results = {
        "ntn": ntn_result,
        "gtn": gtn_result,
    }
    with open(output_base / "combined_results.json", "w") as f:
        json.dump(combined_results, f, indent=2)


if __name__ == "__main__":
    main()
