#!/usr/bin/env python3
"""
Generate experiment config files for CMPO2/CMPO3 image experiments.
"""

import os
import json
import argparse
from pathlib import Path

DATASETS = {
    "MNIST": {"channels": 1, "spatial": 784},
    "FASHION_MNIST": {"channels": 1, "spatial": 784},
    "CIFAR10": {"channels": 3, "spatial": 1024},
    "CIFAR100": {"channels": 3, "spatial": 1024},
}

VALID_N_PATCHES = {
    "MNIST": [4, 7, 14, 28],
    "FASHION_MNIST": [4, 7, 14, 28],
    "CIFAR10": [4, 8, 16, 32],
    "CIFAR100": [4, 8, 16, 32],
}

BASE_TRACKER = {
    "backend": "both",
    "tracker_dir": "experiment_logs",
    "aim_repo": "aim://aimtracking.kosmon.org:443",
}

BASE_OUTPUT = {
    "save_models": False,
    "save_individual_runs": True,
}

CMPO2_GRID = {
    "n_patches": None,
    "n_sites": [2, 3, 4],
    "rank_pixel": [4, 8, 16],
    "rank_patch": [4, 8, 16],
    "lr": [0.01, 0.001, 0.0001],
    "weight_decay": [0.1, 0.01],
    "init_strength": [0.1, 0.01],
}

CMPO3_GRID = {
    "n_patches": None,
    "n_sites": [2, 3, 4],
    "rank_channel": [4, 8],
    "rank_pixel": [4, 8, 16, 32],
    "rank_patch": [4, 8, 16, 32],
    "lr": [0.01, 0.001, 0.0001],
    "weight_decay": [0.1, 0.01],
    "init_strength": [0.1, 0.01],
}

FIXED_PARAMS = {
    "batch_size": 64,
    "n_epochs": 1000,
    "patience": 30,
    "min_delta": 1e-08,
    "optimizer": "adamw",
    "seeds": [42, 7, 123],
}


def generate_config(
    dataset: str,
    model_type: str,
    output_dir: Path,
) -> Path:
    experiment_name = f"{model_type}_{dataset.lower()}"

    grid = CMPO3_GRID.copy() if model_type == "cmpo3" else CMPO2_GRID.copy()
    grid["n_patches"] = VALID_N_PATCHES[dataset]

    config = {
        "experiment_name": experiment_name,
        "dataset": dataset,
        "task": "classification",
    }

    if model_type == "cmpo3":
        config["model_type"] = "cmpo3"

    config["parameter_grid"] = grid
    config["fixed_params"] = FIXED_PARAMS.copy()
    config["tracker"] = BASE_TRACKER.copy()
    config["output"] = {
        **BASE_OUTPUT,
        "results_dir": f"results/{experiment_name}",
    }

    config_path = output_dir / f"{experiment_name}.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    return config_path


def main():
    parser = argparse.ArgumentParser(description="Generate image experiment configs")
    parser.add_argument(
        "--output-dir",
        default="experiments_images/configs",
        help="Output directory for config files",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["cmpo2", "cmpo3"],
        choices=["cmpo2", "cmpo3"],
        help="Model types to generate configs for",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=list(DATASETS.keys()),
        choices=list(DATASETS.keys()),
        help="Datasets to generate configs for",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    configs = []
    for model_type in args.models:
        for dataset in args.datasets:
            if model_type == "cmpo3" and DATASETS[dataset]["channels"] == 1:
                continue

            config_path = generate_config(dataset, model_type, output_dir)
            configs.append(config_path)
            print(f"  Created: {config_path.name}")

    print(f"\nGenerated {len(configs)} config files in {output_dir}")


if __name__ == "__main__":
    main()
