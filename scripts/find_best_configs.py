#!/usr/bin/env python3
"""
Find best config per model × dataset × trainer. Generates Hydra configs for test runs.

Usage: python scripts/find_best_configs.py [--outputs-dir outputs] [--conf-dir conf/best_conf]
"""

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import yaml


@dataclass
class RunResult:
    path: Path
    trainer: str
    model_name: str
    dataset_name: str
    val_quality: float
    best_epoch: int
    success: bool
    singular: bool
    config: dict

    @property
    def L(self) -> int:
        return self.config.get("model", {}).get("L", 0)

    @property
    def bond_dim(self) -> int:
        return self.config.get("model", {}).get("bond_dim", 0)

    @property
    def seed(self) -> int:
        return self.config.get("seed", 0)

    @property
    def init_strength(self) -> float:
        return self.config.get("model", {}).get("init_strength", 0.1)

    @property
    def ridge(self) -> float:
        return self.config.get("trainer", {}).get("ridge", 0.0)

    @property
    def lr(self) -> float:
        return self.config.get("trainer", {}).get("lr", 0.0)

    @property
    def config_key(self) -> tuple:
        """Key identifying the config (excluding seed)."""
        return (self.trainer, self.model_name, self.dataset_name, self.L, self.bond_dim)


@dataclass
class AveragedConfig:
    """Config with val_quality averaged over seeds."""
    trainer: str
    model_name: str
    dataset_name: str
    L: int
    bond_dim: int
    avg_val_quality: float
    std_val_quality: float
    n_seeds: int
    seeds: list[int]
    sample_result: RunResult


def parse_results_json(path: Path) -> Optional[RunResult]:
    try:
        with open(path) as f:
            data = json.load(f)

        config = data.get("config", {})
        trainer_type = config.get("trainer", {}).get("type", "unknown")
        model_name = config.get("model", {}).get("name", "unknown")
        dataset_name = config.get("dataset", {}).get("name", "unknown")

        return RunResult(
            path=path,
            trainer=trainer_type,
            model_name=model_name,
            dataset_name=dataset_name,
            val_quality=data.get("val_quality", float("-inf")),
            best_epoch=data.get("best_epoch", -1),
            success=data.get("success", False),
            singular=data.get("singular", False),
            config=config,
        )
    except (json.JSONDecodeError, KeyError, OSError) as e:
        print(f"Warning: Failed to parse {path}: {e}")
        return None


def find_all_results(outputs_dir: Path) -> list[RunResult]:
    results = []
    for results_file in outputs_dir.rglob("results.json"):
        result = parse_results_json(results_file)
        if result and result.success and not result.singular:
            results.append(result)
    return results


def group_results(
    results: list[RunResult],
) -> dict[tuple[str, str, str], list[RunResult]]:
    groups: dict[tuple[str, str, str], list[RunResult]] = defaultdict(list)
    for result in results:
        key = (result.trainer, result.model_name, result.dataset_name)
        groups[key].append(result)
    return dict(groups)


def compute_averaged_configs(
    results: list[RunResult],
) -> dict[tuple, AveragedConfig]:
    """Group results by config (excluding seed) and compute average val_quality."""
    by_config: dict[tuple, list[RunResult]] = defaultdict(list)
    for result in results:
        by_config[result.config_key].append(result)

    averaged = {}
    for config_key, runs in by_config.items():
        qualities = [r.val_quality for r in runs]
        avg_q = sum(qualities) / len(qualities)
        std_q = (sum((q - avg_q) ** 2 for q in qualities) / len(qualities)) ** 0.5 if len(qualities) > 1 else 0.0

        averaged[config_key] = AveragedConfig(
            trainer=config_key[0],
            model_name=config_key[1],
            dataset_name=config_key[2],
            L=config_key[3],
            bond_dim=config_key[4],
            avg_val_quality=avg_q,
            std_val_quality=std_q,
            n_seeds=len(runs),
            seeds=sorted(set(r.seed for r in runs)),
            sample_result=runs[0],
        )

    return averaged


def find_best_per_group(
    averaged_configs: dict[tuple, AveragedConfig],
) -> dict[tuple[str, str, str], AveragedConfig]:
    """Find the best config (highest avg val_quality) per (trainer, model, dataset)."""
    by_group: dict[tuple[str, str, str], list[AveragedConfig]] = defaultdict(list)
    for config in averaged_configs.values():
        key = (config.trainer, config.model_name, config.dataset_name)
        by_group[key].append(config)

    return {key: max(configs, key=lambda c: c.avg_val_quality) for key, configs in by_group.items()}


def _build_best_configs_table(model_configs: list[tuple[str, AveragedConfig]]) -> dict:
    return {
        dataset: {
            "L": config.L,
            "bond_dim": config.bond_dim,
            "avg_val_quality": round(config.avg_val_quality, 4),
            "std_val_quality": round(config.std_val_quality, 4),
            "n_seeds": config.n_seeds,
        }
        for dataset, config in model_configs
    }


def _write_model_config(
    model: str,
    model_configs: list[tuple[str, AveragedConfig]],
    trainer: str,
    trainer_dir: Path,
) -> None:
    model_file = trainer_dir / f"{model.lower()}.yaml"

    model_config = {
        "_best_configs": _build_best_configs_table(model_configs),
    }

    with open(model_file, "w") as f:
        yaml.dump(model_config, f, default_flow_style=False, sort_keys=False)

    print(f"Generated: {model_file}")


def _write_summary_config(
    trainer: str,
    all_models: list[str],
    all_datasets: list[str],
    total_configs: int,
    trainer_dir: Path,
) -> None:
    summary_file = trainer_dir / "_all.yaml"
    summary = {
        "@package": "_global_",
        "models": all_models,
        "datasets": all_datasets,
        "total_configs": total_configs,
    }

    with open(summary_file, "w") as f:
        f.write(f"# @package _global_\n")
        f.write(f"# Summary of best {trainer.upper()} configs\n")
        f.write(f"# Generated by scripts/find_best_configs.py\n\n")
        yaml.dump(summary, f, default_flow_style=False, sort_keys=False)

    print(f"Generated summary: {summary_file}")


def generate_hydra_config(
    trainer: str,
    best_configs: dict[tuple[str, str, str], AveragedConfig],
    conf_dir: Path,
) -> None:
    trainer_results = {
        (model, dataset): config
        for (t, model, dataset), config in best_configs.items()
        if t == trainer
    }

    if not trainer_results:
        print(f"No results found for trainer: {trainer}")
        return

    by_model: dict[str, list[tuple[str, AveragedConfig]]] = defaultdict(list)
    for (model, dataset), config in trainer_results.items():
        by_model[model].append((dataset, config))

    trainer_dir = conf_dir / trainer
    trainer_dir.mkdir(parents=True, exist_ok=True)

    for model in sorted(by_model.keys()):
        _write_model_config(model, by_model[model], trainer, trainer_dir)

    all_models = sorted(by_model.keys())
    all_datasets = sorted(
        set(dataset for datasets in by_model.values() for dataset, _ in datasets)
    )
    _write_summary_config(trainer, all_models, all_datasets, len(trainer_results), trainer_dir)


def print_summary(best_configs: dict[tuple[str, str, str], AveragedConfig]) -> None:
    print("\n" + "=" * 80)
    print("BEST CONFIGURATIONS SUMMARY (averaged over seeds)")
    print("=" * 80)

    by_trainer: dict[str, list[tuple[str, str, AveragedConfig]]] = defaultdict(list)
    for (trainer, model, dataset), config in best_configs.items():
        by_trainer[trainer].append((model, dataset, config))

    for trainer in sorted(by_trainer.keys()):
        print(f"\n{'─' * 40}")
        print(f"Trainer: {trainer.upper()}")
        print(f"{'─' * 40}")

        configs = sorted(by_trainer[trainer], key=lambda x: (x[0], x[1]))
        for model, dataset, config in configs:
            print(
                f"  {model:15s} | {dataset:20s} | "
                f"avg={config.avg_val_quality:7.4f} ± {config.std_val_quality:.4f} | "
                f"L={config.L} bd={config.bond_dim:2d} (n={config.n_seeds})"
            )

    print(f"\n{'=' * 80}")
    print(f"Total best configs: {len(best_configs)}")
    print(f"  NTN: {sum(1 for (t, _, _) in best_configs if t == 'ntn')}")
    print(f"  GTN: {sum(1 for (t, _, _) in best_configs if t == 'gtn')}")


def main():
    parser = argparse.ArgumentParser(
        description="Find best configurations from ablation study results"
    )
    parser.add_argument(
        "--outputs-dir",
        type=Path,
        default=Path("outputs"),
        help="Directory containing experiment outputs (default: outputs)",
    )
    parser.add_argument(
        "--conf-dir",
        type=Path,
        default=Path("conf/best_conf"),
        help="Directory to save best config YAML files (default: conf/best_conf)",
    )
    parser.add_argument(
        "--json",
        type=Path,
        default=None,
        help="Also save results as JSON to this file",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress detailed output",
    )

    args = parser.parse_args()

    print(f"Scanning {args.outputs_dir} for results...")
    results = find_all_results(args.outputs_dir)
    print(f"Found {len(results)} successful results")

    averaged_configs = compute_averaged_configs(results)
    print(f"Found {len(averaged_configs)} unique configs (trainer, model, dataset, L, bond_dim)")

    best_configs = find_best_per_group(averaged_configs)
    print(f"Found {len(best_configs)} best configs (one per trainer × model × dataset)")

    if not args.quiet:
        print_summary(best_configs)

    print(f"\nGenerating configs in {args.conf_dir}...")
    for trainer in ["ntn", "gtn"]:
        generate_hydra_config(trainer, best_configs, args.conf_dir)

    if args.json:
        json_output = {}
        for (trainer, model, dataset), config in best_configs.items():
            key = f"{trainer}/{model}/{dataset}"
            json_output[key] = {
                "avg_val_quality": config.avg_val_quality,
                "std_val_quality": config.std_val_quality,
                "n_seeds": config.n_seeds,
                "seeds": config.seeds,
                "L": config.L,
                "bond_dim": config.bond_dim,
                "init_strength": config.sample_result.init_strength,
                "sample_path": str(config.sample_result.path),
            }

        with open(args.json, "w") as f:
            json.dump(json_output, f, indent=2)
        print(f"\nSaved JSON output to: {args.json}")

    print("\nDone!")


if __name__ == "__main__":
    main()
