#!/usr/bin/env python3
"""
Generate experiment config files for all UCI datasets.

Creates separate configs for:
1. Base models (MPO2, MMPO2, TypeI variants)
2. LMPO2 models (with reduction_factor)

Both share the SAME experiment name (ntn_{dataset} or gtn_{dataset}) so they're
grouped together in AIM tracking and results.
"""

import json
import os

UCI_DATASETS = [
    # Regression datasets
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
    # Classification datasets
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

HIGH_FEATURE_DATASETS = [
    "adult",
    "bank",
    "mushrooms",
    "student_dropout",
    "popularity",
    "ai4i",
]

def create_ntn_base_config(dataset_name: str, task: str, include_mmpo2: bool = True) -> dict:
    models = ["MPO2", "MPO2TypeI"]
    if include_mmpo2:
        models.extend(["MMPO2", "MMPO2TypeI"])

    return {
        "experiment_name": f"ntn_{dataset_name}",
        "dataset": dataset_name,
        "task": task,
        "parameter_grid": {
            "model": models,
            "L": [3, 4],
            "bond_dim": [8, 12, 16],
            "jitter_start": [5.0],
        },
        "fixed_params": {
            "batch_size": 512,
            "n_epochs": 100,
            "jitter_decay": 0.25,
            "adaptive_jitter": False,
            "patience": 10,
            "min_delta": 0.00001,
            "train_selection": True,
            "init_strength": 0.01,
            "seeds": [42, 7, 123, 256, 999],
        },
        "tracker": {
            "backend": "both",
            "tracker_dir": "experiment_logs",
            "aim_repo": "aim://aimtracking.kosmon.org:443",
        },
        "output": {
            "results_dir": f"results/ntn_{dataset_name}",
            "save_models": False,
            "save_individual_runs": True,
        },
    }


def create_ntn_lmpo2_config(dataset_name: str, task: str) -> dict:
    return {
        "experiment_name": f"ntn_{dataset_name}",
        "dataset": dataset_name,
        "task": task,
        "parameter_grid": {
            "model": ["LMPO2", "LMPO2TypeI"],
            "L": [3, 4],
            "bond_dim": [8, 12, 16],
            "jitter_start": [5.0],
            "reduction_factor": [0.1, 0.3, 0.5],
        },
        "fixed_params": {
            "batch_size": 1024,
            "n_epochs": 100,
            "jitter_decay": 0.25,
            "adaptive_jitter": False,
            "patience": 10,
            "min_delta": 0.000001,
            "train_selection": True,
            "init_strength": 0.01,
            "bond_dim_mpo": 1,
            "seeds": [42, 7, 123, 256, 999],
        },
        "tracker": {
            "backend": "both",
            "tracker_dir": "experiment_logs",
            "aim_repo": "aim://aimtracking.kosmon.org:443",
        },
        "output": {
            "results_dir": f"results/ntn_{dataset_name}_lmpo2",
            "save_models": False,
            "save_individual_runs": True,
        },
    }


def create_gtn_base_config(dataset_name: str, task: str, include_mmpo2: bool = True) -> dict:
    models = ["MPO2", "MPO2TypeI_GTN"]
    if include_mmpo2:
        models.extend(["MMPO2", "MMPO2TypeI_GTN"])

    return {
        "experiment_name": f"gtn_{dataset_name}",
        "dataset": dataset_name,
        "task": task,
        "parameter_grid": {
            "model": models,
            "L": [3, 4],
            "bond_dim": [8, 12, 16],
            "lr": [0.001, 0.0001],
        },
        "fixed_params": {
            "batch_size": 64,
            "n_epochs": 1000,
            "patience": 40,
            "min_delta": 0.00000001,
            "optimizer": "adamw",
            "init_strength": 0.01,
            "seeds": [42, 7, 123, 256, 999],
        },
        "tracker": {
            "backend": "both",
            "tracker_dir": "experiment_logs",
            "aim_repo": "aim://aimtracking.kosmon.org:443",
        },
        "output": {
            "results_dir": f"results/gtn_{dataset_name}",
            "save_models": False,
            "save_individual_runs": True,
        },
    }


def create_gtn_lmpo2_config(dataset_name: str, task: str) -> dict:
    return {
        "experiment_name": f"gtn_{dataset_name}",
        "dataset": dataset_name,
        "task": task,
        "parameter_grid": {
            "model": ["LMPO2", "LMPO2TypeI_GTN"],
            "L": [3, 4],
            "bond_dim": [8, 12, 16],
            "lr": [0.001, 0.0001],
            "reduction_factor": [0.1, 0.3, 0.5],
        },
        "fixed_params": {
            "batch_size": 64,
            "n_epochs": 1000,
            "patience": 40,
            "min_delta": 0.00000001,
            "optimizer": "adamw",
            "init_strength": 0.01,
            "bond_dim_mpo": 2,
            "seeds": [42, 7, 123, 256, 999],
        },
        "tracker": {
            "backend": "both",
            "tracker_dir": "experiment_logs",
            "aim_repo": "aim://aimtracking.kosmon.org:443",
        },
        "output": {
            "results_dir": f"results/gtn_{dataset_name}_lmpo2",
            "save_models": False,
            "save_individual_runs": True,
        },
    }


def main():
    output_dir = os.path.dirname(os.path.abspath(__file__))

    configs_created = []

    for dataset_name, _, task in UCI_DATASETS:
        include_mmpo2 = dataset_name not in HIGH_FEATURE_DATASETS

        for name, config in [
            (
                f"uci_ntn_{dataset_name}.json",
                create_ntn_base_config(dataset_name, task, include_mmpo2),
            ),
            (f"uci_ntn_{dataset_name}_lmpo2.json", create_ntn_lmpo2_config(dataset_name, task)),
            (
                f"uci_gtn_{dataset_name}.json",
                create_gtn_base_config(dataset_name, task, include_mmpo2),
            ),
            (f"uci_gtn_{dataset_name}_lmpo2.json", create_gtn_lmpo2_config(dataset_name, task)),
        ]:
            with open(os.path.join(output_dir, name), "w") as f:
                json.dump(config, f, indent=2)
            configs_created.append(name)
            print(f"Created: {name}")

    print(f"\n{'=' * 60}")
    print(f"Total: {len(configs_created)} config files")
    print(f"Datasets with MMPO2 excluded: {HIGH_FEATURE_DATASETS}")


if __name__ == "__main__":
    main()
