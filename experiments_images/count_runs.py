#!/usr/bin/env python3
"""
Count total experiment runs across all config files.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from experiments.config_parser import expand_parameter_grid


def count_runs_for_config(config_path: Path) -> dict:
    """Count runs for a single config file."""
    with open(config_path) as f:
        config = json.load(f)

    grid_combinations = expand_parameter_grid(config["parameter_grid"])
    seeds = config["fixed_params"].get("seeds", [0])
    if not isinstance(seeds, list):
        seeds = [seeds]

    n_grid = len(grid_combinations)
    n_seeds = len(seeds)
    n_runs = n_grid * n_seeds

    return {
        "config": config_path.name,
        "experiment_name": config["experiment_name"],
        "dataset": config["dataset"],
        "model_type": config.get("model_type", "cmpo2"),
        "grid_size": n_grid,
        "n_seeds": n_seeds,
        "total_runs": n_runs,
        "grid_params": {
            k: len(v) if isinstance(v, list) else 1 for k, v in config["parameter_grid"].items()
        },
    }


def main():
    configs_dir = Path(__file__).parent / "configs"
    config_files = sorted(configs_dir.glob("cmpo*.json"))

    config_files = [f for f in config_files if "test" not in f.name]

    print("=" * 80)
    print("EXPERIMENT RUN COUNT")
    print("=" * 80)

    total_runs = 0
    results = []

    for config_file in config_files:
        info = count_runs_for_config(config_file)
        results.append(info)
        total_runs += info["total_runs"]

        print(f"\n{info['experiment_name']} ({info['model_type'].upper()})")
        print(f"  Dataset: {info['dataset']}")
        print(f"  Grid params: {info['grid_params']}")
        print(f"  Grid combinations: {info['grid_size']}")
        print(f"  Seeds: {info['n_seeds']}")
        print(f"  Total runs: {info['total_runs']:,}")

    print("\n" + "=" * 80)
    print(f"GRAND TOTAL: {total_runs:,} runs across {len(config_files)} configs")
    print("=" * 80)

    return total_runs


if __name__ == "__main__":
    main()
