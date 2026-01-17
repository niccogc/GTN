#!/usr/bin/env python3
"""
Generate experiment config files for all UCI datasets.

This script creates:
1. NTN configs for standard models (MPO2, LMPO2, MMPO2) + TypeI variants
2. GTN configs for gradient-based training

Grid search parameters:
- L (num sites): 2, 3, 4
- bond_dim: 4, 6, 8, 10
- reduction_factor: 0.1, 0.3, 0.5 (applied only to LMPO2/LMPO2TypeI models)
- init_strength: 1.0, 0.1, 0.01
- NTN: jitter_start: 5, 1, 0.1 with decay, epochs: 10, patience: 4
- GTN: weight_decay: 5, 1, 0.1, lr: 0.01, 0.001, 0.0001, epochs: 200, patience: 20

MMPO2 is excluded for datasets with >50 features (cap limit).
"""

import json
import os

# UCI datasets from load_ucirepo.py
# Format: (name, uci_id, task)
UCI_DATASETS = [
    # Regression datasets (11)
    ("student_perf", 320, "regression"),
    ("abalone", 1, "regression"),
    ("obesity", 544, "regression"),
    ("bike", 275, "regression"),
    ("realstate", 477, "regression"),
    ("energy_efficiency", 242, "regression"),
    ("concrete", 165, "regression"),
    ("ai4i", 601, "regression"),
    ("appliances", 374, "regression"),
    ("popularity", 332, "regression"),
    ("seoulBike", 560, "regression"),
    # Classification datasets (10)
    ("iris", 53, "classification"),
    ("hearth", 45, "classification"),
    ("winequalityc", 186, "classification"),
    ("breast", 17, "classification"),
    ("adult", 2, "classification"),
    ("bank", 222, "classification"),
    ("wine", 109, "classification"),
    ("car_evaluation", 19, "classification"),
    ("student_dropout", 697, "classification"),
    ("mushrooms", 73, "classification"),
]

# Datasets with >50 features after one-hot encoding (exclude MMPO2)
# These are approximations based on typical feature counts
HIGH_FEATURE_DATASETS = [
    "adult",  # Many categorical features
    "bank",  # Many categorical features
    "mushrooms",  # Many categorical features
    "student_dropout",  # Large feature set
    "popularity",  # Many features
    "ai4i",  # Industrial dataset with many features
]


def create_ntn_config(dataset_name: str, task: str, include_mmpo2: bool = True) -> dict:
    models = ["MPO2", "LMPO2", "MPO2TypeI", "LMPO2TypeI"]
    if include_mmpo2:
        models.extend(["MMPO2", "MMPO2TypeI"])

    config = {
        "experiment_name": f"{dataset_name}_ntn_grid",
        "dataset": dataset_name,
        "task": task,
        "parameter_grid": {
            "model": models,
            "L": [2, 3, 4],
            "bond_dim": [4, 6, 8, 10],
            "jitter_start": [5.0, 1.0, 0.1],
            "reduction_factor": [0.1, 0.3, 0.5],
            "init_strength": [1.0, 0.1, 0.01],
        },
        "fixed_params": {
            "output_site": 1,
            "batch_size": 100,
            "n_epochs": 10,
            "jitter_decay": 0.95,
            "jitter_min": 0.001,
            "adaptive_jitter": True,
            "patience": 4,
            "min_delta": 0.001,
            "train_selection": True,
            "rank": 5,
            "seeds": [42, 7, 123, 256, 999],
        },
        "tracker": {
            "backend": "both",
            "tracker_dir": "experiment_logs",
            "aim_repo": "aim://aimtracking.kosmon.org:443",
        },
        "output": {
            "results_dir": f"results/{dataset_name}_ntn_grid",
            "save_models": False,
            "save_individual_runs": True,
        },
    }

    return config


def create_gtn_config(dataset_name: str, task: str, include_mmpo2: bool = True) -> dict:
    models = ["MPO2", "LMPO2", "MMPO2", "MPO2TypeI_GTN", "LMPO2TypeI_GTN", "MMPO2TypeI_GTN"]

    config = {
        "experiment_name": f"{dataset_name}_gtn_grid",
        "dataset": dataset_name,
        "task": task,
        "parameter_grid": {
            "model": models,
            "L": [2, 3, 4],
            "bond_dim": [4, 6, 8, 10],
            "weight_decay": [5.0, 1.0, 0.1],
            "lr": [0.01, 0.001, 0.0001],
            "reduction_factor": [0.1, 0.3, 0.5],
            "init_strength": [1.0, 0.1, 0.01],
        },
        "fixed_params": {
            "output_site": 1,
            "batch_size": 32,
            "n_epochs": 200,
            "patience": 20,
            "optimizer": "adamw",
            "rank": 5,
            "seeds": [42, 7, 123, 256, 999],
        },
        "tracker": {
            "backend": "both",
            "tracker_dir": "experiment_logs",
            "aim_repo": "aim://aimtracking.kosmon.org:443",
        },
        "output": {
            "results_dir": f"results/{dataset_name}_gtn_grid",
            "save_models": False,
            "save_individual_runs": True,
        },
    }

    return config


def main():
    """Generate all config files."""

    output_dir = os.path.dirname(os.path.abspath(__file__))

    ntn_configs = []
    gtn_configs = []

    for dataset_name, _, task in UCI_DATASETS:
        include_mmpo2 = dataset_name not in HIGH_FEATURE_DATASETS

        ntn_config = create_ntn_config(dataset_name, task, include_mmpo2)
        ntn_filename = f"uci_ntn_{dataset_name}.json"
        ntn_filepath = os.path.join(output_dir, ntn_filename)

        with open(ntn_filepath, "w") as f:
            json.dump(ntn_config, f, indent=2)

        ntn_configs.append(ntn_filename)
        print(f"Created: {ntn_filename}")

        gtn_config = create_gtn_config(dataset_name, task, include_mmpo2=True)
        gtn_filename = f"uci_gtn_{dataset_name}.json"
        gtn_filepath = os.path.join(output_dir, gtn_filename)

        with open(gtn_filepath, "w") as f:
            json.dump(gtn_config, f, indent=2)

        gtn_configs.append(gtn_filename)
        print(f"Created: {gtn_filename}")

    print(f"\n{'=' * 60}")
    print(f"Generated {len(ntn_configs)} NTN configs")
    print(f"Generated {len(gtn_configs)} GTN configs")
    print(f"Total: {len(ntn_configs) + len(gtn_configs)} config files")
    print(f"\nDatasets with MMPO2 excluded in NTN (>50 features): {HIGH_FEATURE_DATASETS}")
    print(
        f"\nNote: reduction_factor is now included in main configs (only applied to LMPO2/LMPO2TypeI models)"
    )


if __name__ == "__main__":
    main()
