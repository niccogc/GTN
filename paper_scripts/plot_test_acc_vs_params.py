#!/usr/bin/env python3
import argparse
import json
import re
import sys
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np

# Paths relative to project root (parent of paper_scripts/)
PROJECT_ROOT = Path(__file__).parent.parent
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

MNIST_DIR = OUTPUTS_DIR / "gtn" / "MNIST" / "CMPO2"
FASHION_DIR = OUTPUTS_DIR / "gtn" / "FASHION_MNIST" / "CMPO2"
CNN_MNIST_DIR = OUTPUTS_DIR / "cnn" / "MNIST" / "BaselineCNN_rg0_init0"
CNN_FASHION_DIR = OUTPUTS_DIR / "cnn" / "FASHION_MNIST" / "BaselineCNN_rg0_init0"
NTN_MNIST_DIR = OUTPUTS_DIR / "ntn" / "MNIST" / "CMPO2_rg5_init0.1"
NTN_FASHION_DIR = OUTPUTS_DIR / "ntn" / "FASHION_MNIST" / "CMPO2_rg5_init0.1"
OUTPUT_PATH = Path(__file__).parent / "images" / "test_acc_vs_params.pdf"
OUTPUT_PATH_NTN = Path(__file__).parent / "images" / "test_acc_vs_params_with_ntn.pdf"

cpd_mnist = [
    ('CPD', 2, 47110, 92.79),
    ('CPD', 5, 117760, 95.30),
    ('CPD', 10, 235510, 96.26),
    ('CPD', 20, 471010, 96.92),
    ('CPD', 50, 1177510, 97.44),
    ('CPD', 150, 3532510, 97.77),
]

cpd_fashion = [
    ('CPD', 2, 47110, 84.59),
    ('CPD', 5, 117760, 86.01),
    ('CPD', 10, 235510, 86.52),
    ('CPD', 20, 471010, 87.07),
    ('CPD', 50, 1177510, 87.40),
    ('CPD', 150, 3532510, 87.92),
]

tnml_mnist = [
    ('TNML', 10, 155720, 98.15),
    ('TNML', 20, 621192, 97.55),
    ('TNML', 40, 2478280, 98.35),
    ('TNML', 80, 9887432, 98.14),
]

tnml_fashion = [
    ('TNML', 10, 155720, 87.49),
    ('TNML', 20, 621192, 87.85),
    ('TNML', 40, 2478280, 88.00),
    ('TNML', 80, 9887432, 87.49),
]


def load_cmpo2_results(base_dir, max_std=2.0):
    """Load CMPO2 results, filtering out configs with high variance (instabilities)."""
    results_by_config = defaultdict(list)
    
    for config_dir in base_dir.iterdir():
        if not config_dir.is_dir():
            continue
        results_file = config_dir / "results.json"
        if not results_file.exists():
            continue
        
        with open(results_file) as f:
            data = json.load(f)
        
        n_params = data.get("n_parameters")
        test_acc = data.get("test_accuracy")
        
        if n_params is None or test_acc is None:
            continue
        
        config_key = config_dir.name.rsplit("_seed", 1)[0]
        results_by_config[config_key].append((n_params, test_acc * 100))
    
    aggregated = []
    for config_key, runs in results_by_config.items():
        n_params = runs[0][0]
        accs = [r[1] for r in runs]
        mean_acc = np.mean(accs)
        std_acc = np.std(accs) if len(accs) > 1 else 0
        # Filter out results with large std (instabilities)
        if std_acc <= max_std:
            aggregated.append((n_params, mean_acc, std_acc))
    
    return aggregated


def load_cnn_results(base_dir, max_std=2.0):
    """Load CNN (BaselineCNN) results, aggregating by config across seeds."""
    if not base_dir.exists():
        return []
    
    results_by_config = defaultdict(list)
    
    for config_dir in base_dir.iterdir():
        if not config_dir.is_dir():
            continue
        results_file = config_dir / "results.json"
        if not results_file.exists():
            continue
        
        with open(results_file) as f:
            data = json.load(f)
        
        n_params = data.get("n_parameters")
        # CNN uses test_quality (0-1 scale), not test_accuracy
        test_acc = data.get("test_quality")
        
        if n_params is None or test_acc is None:
            continue
        
        # Extract config key by removing seed suffix (e.g., "nlayers1_ch2_fc0_seed42" -> "nlayers1_ch2_fc0")
        config_key = config_dir.name.rsplit("_seed", 1)[0]
        results_by_config[config_key].append((n_params, test_acc * 100))
    
    aggregated = []
    for config_key, runs in results_by_config.items():
        n_params = runs[0][0]
        accs = [r[1] for r in runs]
        mean_acc = np.mean(accs)
        std_acc = np.std(accs) if len(accs) > 1 else 0
        # Filter out results with large std (instabilities)
        if std_acc <= max_std:
            aggregated.append((n_params, mean_acc, std_acc))
    
    return aggregated


def compute_cmpo2_n_parameters(config: dict, dataset_info: dict) -> int:
    """Compute n_parameters for CMPO2 model from config."""
    L = config["model"]["L"]
    rank_pixel = config["model"]["rank_pixel"]
    rank_patch = config["model"]["rank_patch"]
    pixel_dim = dataset_info["pixels_per_patch"]
    patch_dim = dataset_info["n_patches"]
    output_dim = dataset_info["n_classes"]
    
    # Import and instantiate the model to count parameters
    sys.path.insert(0, str(PROJECT_ROOT))
    from model.image_models import CMPO2
    
    model = CMPO2(
        L=L,
        pixel_dim=pixel_dim,
        patch_dim=patch_dim,
        pixel_bond_dim=rank_pixel,
        patch_bond_dim=rank_patch,
        output_dim=output_dim,
    )
    
    # Count parameters in the tensor network
    n_params = sum(t.size for t in model.tn)
    return n_params


def load_ntn_cmpo2_results(base_dir, max_std=2.0, min_acc=50.0):
    """Load NTN CMPO2 results, filtering out failed runs and configs with high variance."""
    if not base_dir.exists():
        return []
    
    results_by_config = defaultdict(list)
    
    for config_dir in base_dir.iterdir():
        if not config_dir.is_dir():
            continue
        results_file = config_dir / "results.json"
        if not results_file.exists():
            continue
        
        with open(results_file) as f:
            data = json.load(f)
        
        # NTN uses test_quality (0-1 scale)
        test_acc = data.get("test_quality")
        
        if test_acc is None:
            continue
        
        # Skip failed runs (low accuracy indicates singularity/failure)
        if test_acc * 100 < min_acc:
            continue
        
        # Compute n_parameters from config if not present
        n_params = data.get("n_parameters")
        if n_params is None:
            try:
                n_params = compute_cmpo2_n_parameters(data["config"], data["dataset_info"])
            except Exception:
                continue
        
        config_key = config_dir.name.rsplit("_seed", 1)[0]
        results_by_config[config_key].append((n_params, test_acc * 100))
    
    aggregated = []
    for config_key, runs in results_by_config.items():
        if not runs:
            continue
        n_params = runs[0][0]
        accs = [r[1] for r in runs]
        mean_acc = np.mean(accs)
        std_acc = np.std(accs) if len(accs) > 1 else 0
        # Filter out results with large std among successful runs
        if std_acc <= max_std:
            aggregated.append((n_params, mean_acc, std_acc))
    
    return aggregated


def plot_dataset(ax, cmpo2_data, cpd_data, tnml_data, cnn_data, title, ntn_data=None):
    cmpo2_params = [d[0] for d in cmpo2_data]
    cmpo2_acc = [d[1] for d in cmpo2_data]
    cmpo2_std = [d[2] for d in cmpo2_data]
    
    cpd_params = [d[2] for d in cpd_data]
    cpd_acc = [d[3] for d in cpd_data]
    
    tnml_params = [d[2] for d in tnml_data]
    tnml_acc = [d[3] for d in tnml_data]
    
    ax.errorbar(cmpo2_params, cmpo2_acc, yerr=cmpo2_std, fmt="o", c="#1f77b4", 
                markersize=4, alpha=0.7, label="G-CMPO2", capsize=2, elinewidth=0.8)
    ax.plot(cpd_params, cpd_acc, "s-", color="#ff7f0e", markersize=6, linewidth=1.5, label="TeMPO")
    ax.plot(tnml_params, tnml_acc, "^-", color="#2ca02c", markersize=6, linewidth=1.5, label="TNML-F")
    
    # Plot CNN results if available
    if cnn_data:
        cnn_params = [d[0] for d in cnn_data]
        cnn_acc = [d[1] for d in cnn_data]
        cnn_std = [d[2] for d in cnn_data]
        ax.errorbar(cnn_params, cnn_acc, yerr=cnn_std, fmt="d", c="#d62728", 
                    markersize=4, alpha=0.7, label="CNN + MLP", capsize=2, elinewidth=0.8)
    
    # Plot NTN results if available
    if ntn_data:
        ntn_params = [d[0] for d in ntn_data]
        ntn_acc = [d[1] for d in ntn_data]
        ntn_std = [d[2] for d in ntn_data]
        ax.errorbar(ntn_params, ntn_acc, yerr=ntn_std, fmt="p", c="#9467bd", 
                    markersize=5, alpha=0.7, label="N-CMPO2", capsize=2, elinewidth=0.8)
    
    ax.set_xscale("log")
    ax.set_xlabel("Number of Parameters", fontsize=11)
    ax.set_ylabel("Test Accuracy (%)", fontsize=11)
    ax.set_title(title, fontsize=12)
    ax.grid(True, alpha=0.3)


def main():
    parser = argparse.ArgumentParser(description="Plot test accuracy vs parameters")
    parser.add_argument("--add-ntn", action="store_true", 
                        help="Include N-CMPO2 (NTN) results in the plot")
    args = parser.parse_args()

    mnist_cmpo2 = load_cmpo2_results(MNIST_DIR)
    fashion_cmpo2 = load_cmpo2_results(FASHION_DIR)
    mnist_cnn = load_cnn_results(CNN_MNIST_DIR)
    fashion_cnn = load_cnn_results(CNN_FASHION_DIR)

    # Load NTN results if requested
    mnist_ntn = None
    fashion_ntn = None
    if args.add_ntn:
        mnist_ntn = load_ntn_cmpo2_results(NTN_MNIST_DIR)
        fashion_ntn = load_ntn_cmpo2_results(NTN_FASHION_DIR)
        print(f"Loaded {len(mnist_ntn)} NTN MNIST configs, {len(fashion_ntn)} NTN Fashion configs")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    plot_dataset(ax1, mnist_cmpo2, cpd_mnist, tnml_mnist, mnist_cnn, "MNIST", ntn_data=mnist_ntn)
    plot_dataset(ax2, fashion_cmpo2, cpd_fashion, tnml_fashion, fashion_cnn, "Fashion-MNIST", ntn_data=fashion_ntn)

    # Single shared legend at the bottom - collect from both axes to ensure all labels present
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    # Merge handles/labels, keeping unique labels in order
    label_to_handle = dict(zip(labels1, handles1))
    for h, l in zip(handles2, labels2):
        if l not in label_to_handle:
            label_to_handle[l] = h
    # Preserve a consistent order
    label_order = ["G-CMPO2", "TeMPO", "TNML-F", "CNN + MLP", "N-CMPO2"]
    handles = [label_to_handle[l] for l in label_order if l in label_to_handle]
    labels = [l for l in label_order if l in label_to_handle]
    ncol = len(labels)
    fig.legend(handles, labels, loc="lower center", ncol=ncol, fontsize=10, bbox_to_anchor=(0.5, -0.02))

    plt.tight_layout(rect=[0, 0.05, 1, 1])  # Leave space for legend at bottom
    output_path = OUTPUT_PATH_NTN if args.add_ntn else OUTPUT_PATH
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Plot saved to: {output_path}")
    plt.close()


if __name__ == "__main__":
    main()
