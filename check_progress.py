#!/usr/bin/env python3
"""
Ablation study progress tracker for GTN experiments.

Checks completed runs and displays progress bars and summary statistics.

Usage:
    python check_progress.py              # Default view
    python check_progress.py --verbose    # Show all incomplete experiments
    python check_progress.py --missing    # List all missing run paths
    python check_progress.py --json       # Output as JSON
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

# Configuration from the experiment setup
MODELS = [
    "MPO2",
    "LMPO2",
    "MMPO2",
    "MPO2TypeI",
    "LMPO2TypeI",
    "MMPO2TypeI",
    "CPDA",
    "CPDATypeI",
]
TRAINERS = ["ntn", "gtn"]
SIZES = ["small", "medium", "large"]

CONF_DIR = Path("conf")


def load_datasets_from_conf() -> tuple[list[str], dict[str, str]]:
    """Load dataset names and their sizes from conf/dataset/*.yaml files."""
    import re

    datasets = []
    dataset_sizes = {}
    dataset_dir = CONF_DIR / "dataset"

    if not dataset_dir.exists():
        # Fallback if conf dir not found
        print(
            "Warning: conf/dataset directory not found, using defaults", file=sys.stderr
        )
        return _get_fallback_datasets()

    for yaml_file in sorted(dataset_dir.glob("*.yaml")):
        name = yaml_file.stem
        # Skip base config and size configs
        if name.startswith("_"):
            continue

        datasets.append(name)

        # Parse the yaml file to find size reference
        try:
            content = yaml_file.read_text()
            # Look for "- size/small", "- size/medium", or "- size/large"
            match = re.search(r"-\s*size/(\w+)", content)
            if match:
                dataset_sizes[name] = match.group(1)
            else:
                dataset_sizes[name] = "unknown"
        except Exception:
            dataset_sizes[name] = "unknown"

    return datasets, dataset_sizes


def _get_fallback_datasets() -> tuple[list[str], dict[str, str]]:
    """Fallback dataset list if conf files can't be read."""
    datasets = [
        "abalone",
        "adult",
        "ai4i",
        "appliances",
        "bank",
        "bike",
        "breast",
        "car_evaluation",
        "concrete",
        "energy_efficiency",
        "hearth",
        "iris",
        "mushrooms",
        "obesity",
        "popularity",
        "realstate",
        "seoulBike",
        "student_dropout",
        "student_perf",
        "wine",
        "winequalityc",
    ]
    # Default all to unknown if we can't read configs
    dataset_sizes = {d: "unknown" for d in datasets}
    return datasets, dataset_sizes


# Load datasets and sizes from config files
DATASETS, DATASET_SIZES = load_datasets_from_conf()


def load_trainer_config(trainer: str) -> dict[str, float | None]:
    """Load trainer configuration from conf/trainer/{trainer}.yaml."""
    import re

    trainer_file = CONF_DIR / "trainer" / f"{trainer}.yaml"
    config: dict[str, float | None] = {"ridge": None}

    if not trainer_file.exists():
        return config

    try:
        content = trainer_file.read_text()
        # Parse ridge value
        match = re.search(r"^\s*ridge:\s*([\d.]+)", content, re.MULTILINE)
        if match:
            config["ridge"] = float(match.group(1))
    except Exception:
        pass

    return config


def load_model_base_config() -> dict[str, float | None]:
    """Load base model configuration from conf/model/_base.yaml."""
    import re

    base_file = CONF_DIR / "model" / "_base.yaml"
    config: dict[str, float | None] = {"init_strength": None}

    if not base_file.exists():
        return config

    try:
        content = base_file.read_text()
        # Parse init_strength value
        match = re.search(r"^\s*init_strength:\s*([\d.]+)", content, re.MULTILINE)
        if match:
            config["init_strength"] = float(match.group(1))
    except Exception:
        pass

    return config


def load_sweep_params_from_conf() -> dict:
    """Load sweep parameters from conf/experiment/_base.yaml."""
    import re

    base_file = CONF_DIR / "experiment" / "_base.yaml"
    params = {
        "L_values": [3, 4],
        "bond_dim_values": [4, 8, 12],
        "seeds": [42, 10090, 32874, 47311, 47303],
    }

    if not base_file.exists():
        return params

    try:
        content = base_file.read_text()
        # Parse model.L values
        match = re.search(r"model\.L:\s*([\d,]+)", content)
        if match:
            params["L_values"] = [int(x.strip()) for x in match.group(1).split(",")]
        # Parse model.bond_dim values
        match = re.search(r"model\.bond_dim:\s*([\d,]+)", content)
        if match:
            params["bond_dim_values"] = [
                int(x.strip()) for x in match.group(1).split(",")
            ]
        # Parse seed values
        match = re.search(r"seed:\s*([\d,]+)", content)
        if match:
            params["seeds"] = [int(x.strip()) for x in match.group(1).split(",")]
    except Exception:
        pass

    return params


def load_lmpo2_config() -> dict:
    """Load LMPO2 model configuration from conf/model/lmpo2.yaml."""
    import re

    filepath = CONF_DIR / "model" / "lmpo2.yaml"
    config = {
        "reduction_factors": [0.3, 0.5, 0.9],
        "bond_dim_mpo": 1,
    }

    if not filepath.exists():
        return config

    try:
        content = filepath.read_text()
        # Parse reduction_factor sweep values
        match = re.search(r"model\.reduction_factor:\s*([\d.,]+)", content)
        if match:
            config["reduction_factors"] = [
                float(x.strip()) for x in match.group(1).split(",")
            ]
        # Parse bond_dim_mpo
        match = re.search(r"^\s*bond_dim_mpo:\s*(\d+)", content, re.MULTILINE)
        if match:
            config["bond_dim_mpo"] = int(match.group(1))
    except Exception:
        pass

    return config


def load_cpda_config() -> dict:
    """Load CPDA experiment configuration from conf/experiment/cpda_gtn_sweep.yaml."""
    import re

    filepath = CONF_DIR / "experiment" / "cpda_gtn_sweep.yaml"
    config = {
        "bond_dim_values": [8, 16, 32, 64],  # Default CPDA bond dimensions
    }

    if not filepath.exists():
        return config

    try:
        content = filepath.read_text()
        # Parse model.bond_dim values
        match = re.search(r"model\.bond_dim:\s*([\d,]+)", content)
        if match:
            config["bond_dim_values"] = [
                int(x.strip()) for x in match.group(1).split(",")
            ]
    except Exception:
        pass

    return config


# Load all config values
_SWEEP_PARAMS = load_sweep_params_from_conf()
L_VALUES = _SWEEP_PARAMS["L_values"]
BOND_DIM_VALUES = _SWEEP_PARAMS["bond_dim_values"]
SEEDS = _SWEEP_PARAMS["seeds"]

# Load LMPO2 model config (bond_dim_mpo is always 1 for all LMPO2 variants)
_LMPO2_CONFIG = load_lmpo2_config()
REDUCTION_FACTORS = _LMPO2_CONFIG["reduction_factors"]
BOND_DIM_MPO = _LMPO2_CONFIG["bond_dim_mpo"]

# Load CPDA config (different bond dimensions)
_CPDA_CONFIG = load_cpda_config()
BOND_DIM_CPDA = _CPDA_CONFIG["bond_dim_values"]

# Load trainer ridge values
RIDGE_VALUES = {
    "ntn": load_trainer_config("ntn").get("ridge", 5),
    "gtn": load_trainer_config("gtn").get("ridge", 0.005),
}
INIT_STRENGTH = load_model_base_config().get("init_strength", 0.1)

OUTPUTS_DIR = Path("outputs")


def _format_number(value) -> str:
    """Format number as Hydra does - integers without decimal, floats with."""
    if value is None:
        return "0"
    if float(value) == int(value):
        return str(int(value))
    return str(value)


def get_experiment_path(model: str, dataset: str, trainer: str) -> Path:
    """Generate experiment directory path as Hydra does.

    Format: outputs/{trainer}/{dataset}/{model}_rg{ridge}_init{init_strength}
    """
    ridge = _format_number(RIDGE_VALUES[trainer])
    init = _format_number(INIT_STRENGTH)
    model_dir = f"{model}_rg{ridge}_init{init}"
    return OUTPUTS_DIR / trainer / dataset / model_dir


def get_expected_subdirs(model: str) -> list[str]:
    """Get expected subdirectory names for a given model."""
    subdirs = []
    is_lmpo2 = "LMPO2" in model
    is_cpda = "CPDA" in model

    # Use CPDA-specific bond dimensions if applicable
    bond_dims = BOND_DIM_CPDA if is_cpda else BOND_DIM_VALUES

    for L in L_VALUES:
        for bd in bond_dims:
            for seed in SEEDS:
                if is_lmpo2:
                    for rf in REDUCTION_FACTORS:
                        subdirs.append(
                            f"L{L}_bd{bd}_seed{seed}_rf{rf}_bondmpo{BOND_DIM_MPO}"
                        )
                else:
                    subdirs.append(f"L{L}_bd{bd}_seed{seed}")
    return subdirs


def count_runs_for_model(model: str) -> int:
    """Get the expected number of runs for a model."""
    is_cpda = "CPDA" in model
    bond_dims = BOND_DIM_CPDA if is_cpda else BOND_DIM_VALUES
    base_runs = len(L_VALUES) * len(bond_dims) * len(SEEDS)
    if "LMPO2" in model:
        return base_runs * len(REDUCTION_FACTORS)
    return base_runs


def check_result(result_path: Path) -> dict:
    """Check a results.json file and return status info."""
    if not result_path.exists():
        return {
            "status": "missing",
            "success": None,
            "singular": None,
            "oom_error": None,
        }

    try:
        with open(result_path) as f:
            data = json.load(f)
        return {
            "status": "completed",
            "success": data.get("success", False),
            "singular": data.get("singular", False),
            "oom_error": data.get("oom_error", False),
            "val_quality": data.get("val_quality"),
        }
    except (json.JSONDecodeError, KeyError):
        return {
            "status": "corrupted",
            "success": None,
            "singular": None,
            "oom_error": None,
        }


def print_progress_bar(
    completed: int, total: int, width: int = 40, prefix: str = ""
) -> str:
    """Generate a progress bar string."""
    if total == 0:
        pct = 0
        filled = 0
    else:
        pct = completed / total * 100
        filled = int(width * completed / total)

    bar = "█" * filled + "░" * (width - filled)
    return f"{prefix}[{bar}] {completed:4d}/{total:4d} ({pct:5.1f}%)"


def collect_stats():
    """Collect all statistics from the outputs directory."""
    stats = {
        "by_model": defaultdict(
            lambda: {
                "done": 0,
                "total": 0,
                "success": 0,
                "failed": 0,
                "singular": 0,
                "oom": 0,
            }
        ),
        "by_dataset": defaultdict(
            lambda: {
                "done": 0,
                "total": 0,
                "success": 0,
                "failed": 0,
                "singular": 0,
                "oom": 0,
            }
        ),
        "by_trainer": defaultdict(
            lambda: {
                "done": 0,
                "total": 0,
                "success": 0,
                "failed": 0,
                "singular": 0,
                "oom": 0,
            }
        ),
        "by_model_trainer": defaultdict(
            lambda: {
                "done": 0,
                "total": 0,
                "success": 0,
                "failed": 0,
                "singular": 0,
                "oom": 0,
            }
        ),
        "by_size": defaultdict(
            lambda: {
                "done": 0,
                "total": 0,
                "success": 0,
                "failed": 0,
                "singular": 0,
                "oom": 0,
            }
        ),
        "by_size_trainer": defaultdict(
            lambda: {
                "done": 0,
                "total": 0,
                "success": 0,
                "failed": 0,
                "singular": 0,
                "oom": 0,
            }
        ),
        "by_combo": {},  # (model, dataset, trainer) -> details
    }

    missing_runs = []
    failed_runs = []
    singular_runs = []
    oom_runs = []

    total_expected = 0
    total_completed = 0
    total_success = 0
    total_singular = 0
    total_oom = 0
    total_failed = 0

    # Iterate through all expected combinations
    for model in MODELS:
        expected_runs = count_runs_for_model(model)
        expected_subdirs = get_expected_subdirs(model)

        for dataset in DATASETS:
            for trainer in TRAINERS:
                exp_dir = get_experiment_path(model, dataset, trainer)
                ridge = RIDGE_VALUES[trainer]
                exp_path_str = (
                    f"{trainer}/{dataset}/{model}_rg{ridge}_init{INIT_STRENGTH}"
                )

                combo_key = (model, dataset, trainer)
                model_trainer_key = (model, trainer)
                size = DATASET_SIZES.get(dataset, "unknown")
                size_trainer_key = (size, trainer)
                combo_stats = {
                    "done": 0,
                    "total": expected_runs,
                    "success": 0,
                    "failed": 0,
                    "singular": 0,
                    "oom": 0,
                    "missing": [],
                }

                for subdir in expected_subdirs:
                    result_path = exp_dir / subdir / "results.json"
                    result = check_result(result_path)

                    total_expected += 1
                    stats["by_model"][model]["total"] += 1
                    stats["by_dataset"][dataset]["total"] += 1
                    stats["by_trainer"][trainer]["total"] += 1
                    stats["by_model_trainer"][model_trainer_key]["total"] += 1
                    stats["by_size"][size]["total"] += 1
                    stats["by_size_trainer"][size_trainer_key]["total"] += 1

                    if result["status"] == "completed":
                        total_completed += 1
                        combo_stats["done"] += 1
                        stats["by_model"][model]["done"] += 1
                        stats["by_dataset"][dataset]["done"] += 1
                        stats["by_trainer"][trainer]["done"] += 1
                        stats["by_model_trainer"][model_trainer_key]["done"] += 1
                        stats["by_size"][size]["done"] += 1
                        stats["by_size_trainer"][size_trainer_key]["done"] += 1

                        if result["success"]:
                            total_success += 1
                            combo_stats["success"] += 1
                            stats["by_model"][model]["success"] += 1
                            stats["by_dataset"][dataset]["success"] += 1
                            stats["by_trainer"][trainer]["success"] += 1
                            stats["by_model_trainer"][model_trainer_key]["success"] += 1
                            stats["by_size"][size]["success"] += 1
                            stats["by_size_trainer"][size_trainer_key]["success"] += 1
                        elif result["singular"]:
                            total_singular += 1
                            combo_stats["singular"] += 1
                            stats["by_model"][model]["singular"] += 1
                            stats["by_dataset"][dataset]["singular"] += 1
                            stats["by_trainer"][trainer]["singular"] += 1
                            stats["by_model_trainer"][model_trainer_key][
                                "singular"
                            ] += 1
                            stats["by_size"][size]["singular"] += 1
                            stats["by_size_trainer"][size_trainer_key]["singular"] += 1
                            singular_runs.append(f"{exp_path_str}/{subdir}")
                        elif result.get("oom_error"):
                            total_oom += 1
                            combo_stats["oom"] += 1
                            stats["by_model"][model]["oom"] += 1
                            stats["by_dataset"][dataset]["oom"] += 1
                            stats["by_trainer"][trainer]["oom"] += 1
                            stats["by_model_trainer"][model_trainer_key]["oom"] += 1
                            stats["by_size"][size]["oom"] += 1
                            stats["by_size_trainer"][size_trainer_key]["oom"] += 1
                            oom_runs.append(f"{exp_path_str}/{subdir}")
                        else:
                            total_failed += 1
                            combo_stats["failed"] += 1
                            stats["by_model"][model]["failed"] += 1
                            stats["by_dataset"][dataset]["failed"] += 1
                            stats["by_trainer"][trainer]["failed"] += 1
                            stats["by_model_trainer"][model_trainer_key]["failed"] += 1
                            stats["by_size"][size]["failed"] += 1
                            stats["by_size_trainer"][size_trainer_key]["failed"] += 1
                            failed_runs.append(f"{exp_path_str}/{subdir}")
                    else:
                        combo_stats["missing"].append(subdir)
                        missing_runs.append(f"{exp_path_str}/{subdir}")

                stats["by_combo"][combo_key] = combo_stats

    return {
        "stats": stats,
        "missing_runs": missing_runs,
        "failed_runs": failed_runs,
        "singular_runs": singular_runs,
        "oom_runs": oom_runs,
        "totals": {
            "expected": total_expected,
            "completed": total_completed,
            "success": total_success,
            "singular": total_singular,
            "oom": total_oom,
            "failed": total_failed,
        },
    }


def generate_report(data: dict, verbose: bool = False) -> str:
    """Generate the progress report as a string."""
    lines = []

    def out(text: str = ""):
        lines.append(text)

    stats = data["stats"]
    missing_runs = data["missing_runs"]
    failed_runs = data["failed_runs"]
    singular_runs = data["singular_runs"]
    oom_runs = data.get("oom_runs", [])
    totals = data["totals"]

    total_expected = totals["expected"]
    total_completed = totals["completed"]
    total_success = totals["success"]
    total_singular = totals["singular"]
    total_oom = totals.get("oom", 0)
    total_failed = totals["failed"]

    # Print report
    out("=" * 80)
    out("                     GTN/NTN ABLATION STUDY PROGRESS")
    out("=" * 80)
    out()

    # Overall progress
    out("OVERALL PROGRESS")
    out("-" * 80)
    out(print_progress_bar(total_completed, total_expected, prefix="Total:     "))
    out(
        f"  Success: {total_success:4d}  |  Singular: {total_singular:4d}  |  OOM: {total_oom:4d}  |  Failed: {total_failed:4d}  |  Missing: {total_expected - total_completed:4d}"
    )
    out()

    # By trainer
    out("BY TRAINER")
    out("-" * 80)
    for trainer in TRAINERS:
        s = stats["by_trainer"][trainer]
        out(
            print_progress_bar(
                s["done"], s["total"], prefix=f"{trainer.upper():5s}:     "
            )
        )
        out(
            f"  Success: {s['success']:4d}  |  Singular: {s['singular']:4d}  |  OOM: {s['oom']:4d}  |  Failed: {s['failed']:4d}"
        )
    out()

    # Model x Trainer matrix
    out("MODEL x TRAINER MATRIX")
    out("-" * 80)
    out(f"{'Model':<12s} | {'NTN':^20s} | {'GTN':^20s}")
    out("-" * 80)
    for model in MODELS:
        ntn_s = stats["by_model_trainer"][(model, "ntn")]
        gtn_s = stats["by_model_trainer"][(model, "gtn")]
        ntn_pct = ntn_s["done"] / ntn_s["total"] * 100 if ntn_s["total"] > 0 else 0
        gtn_pct = gtn_s["done"] / gtn_s["total"] * 100 if gtn_s["total"] > 0 else 0
        ntn_str = f"{ntn_s['done']:4d}/{ntn_s['total']:4d} ({ntn_pct:5.1f}%)"
        gtn_str = f"{gtn_s['done']:4d}/{gtn_s['total']:4d} ({gtn_pct:5.1f}%)"
        out(f"{model:<12s} | {ntn_str:^20s} | {gtn_str:^20s}")
    out()

    # Size x Trainer matrix (WHAT TO RUN)
    out("BY DATASET SIZE (for job submission)")
    out("-" * 80)
    out(f"{'Size':<8s} | {'NTN':^24s} | {'GTN':^24s} | {'Total Missing':^14s}")
    out("-" * 80)
    for size in SIZES:
        ntn_s = stats["by_size_trainer"][(size, "ntn")]
        gtn_s = stats["by_size_trainer"][(size, "gtn")]
        ntn_missing = ntn_s["total"] - ntn_s["done"]
        gtn_missing = gtn_s["total"] - gtn_s["done"]
        total_missing = ntn_missing + gtn_missing
        ntn_pct = ntn_s["done"] / ntn_s["total"] * 100 if ntn_s["total"] > 0 else 0
        gtn_pct = gtn_s["done"] / gtn_s["total"] * 100 if gtn_s["total"] > 0 else 0
        ntn_str = f"{ntn_s['done']:4d}/{ntn_s['total']:4d} ({ntn_pct:5.1f}%)"
        gtn_str = f"{gtn_s['done']:4d}/{gtn_s['total']:4d} ({gtn_pct:5.1f}%)"
        out(f"{size:<8s} | {ntn_str:^24s} | {gtn_str:^24s} | {total_missing:^14d}")
    out()

    # By dataset
    out("BY DATASET")
    out("-" * 80)
    complete_datasets = []
    incomplete_datasets = []
    for dataset in DATASETS:
        s = stats["by_dataset"][dataset]
        if s["done"] == s["total"]:
            complete_datasets.append(dataset)
        else:
            incomplete_datasets.append((dataset, s))

    if complete_datasets:
        out(f"  COMPLETE: {', '.join(complete_datasets)}")
        out()

    for dataset, s in incomplete_datasets:
        label = f"{dataset[:15]:15s}"
        out(print_progress_bar(s["done"], s["total"], width=30, prefix=f"{label}: "))
    out()

    # Detailed combo status (incomplete ones)
    incomplete = [
        (k, v) for k, v in stats["by_combo"].items() if v["done"] < v["total"]
    ]
    if incomplete:
        out("INCOMPLETE EXPERIMENTS")
        out("-" * 80)
        incomplete.sort(
            key=lambda x: x[1]["done"] / max(x[1]["total"], 1), reverse=True
        )
        show_count = len(incomplete) if verbose else 20
        for (model, dataset, trainer), s in incomplete[:show_count]:
            pct = s["done"] / s["total"] * 100 if s["total"] > 0 else 0
            out(
                f"  {model:12s} | {dataset:16s} | {trainer:3s} | {s['done']:3d}/{s['total']:3d} ({pct:5.1f}%)"
            )
        if not verbose and len(incomplete) > 20:
            out(f"  ... and {len(incomplete) - 20} more (use --verbose to see all)")
    out()

    # Summary of issues
    if singular_runs:
        out(f"SINGULAR MATRIX FAILURES: {len(singular_runs)}")
        out("-" * 80)
        # Group by experiment
        singular_by_exp = defaultdict(int)
        for run in singular_runs:
            exp = run.split("/")[0]
            singular_by_exp[exp] += 1
        for exp, count in sorted(singular_by_exp.items(), key=lambda x: -x[1])[:10]:
            out(f"  {exp}: {count}")
        out()

    if oom_runs:
        out(f"CUDA OUT OF MEMORY FAILURES: {len(oom_runs)}")
        out("-" * 80)
        # Group by experiment
        oom_by_exp = defaultdict(int)
        for run in oom_runs:
            exp = run.split("/")[0]
            oom_by_exp[exp] += 1
        for exp, count in sorted(oom_by_exp.items(), key=lambda x: -x[1])[:10]:
            out(f"  {exp}: {count}")
        out()

    if failed_runs:
        out(f"OTHER FAILURES: {len(failed_runs)}")
        out("-" * 80)
        for run in failed_runs[:10]:
            out(f"  {run}")
        if len(failed_runs) > 10:
            out(f"  ... and {len(failed_runs) - 10} more")
        out()

    # What's left to do
    out("=" * 80)
    out("REMAINING WORK SUMMARY")
    out("=" * 80)
    remaining = total_expected - total_completed
    out(f"  Runs remaining: {remaining}")
    out(f"  Experiments with missing runs: {len(incomplete)}")

    # Group missing by trainer
    missing_ntn = sum(1 for r in missing_runs if "_ntn_" in r)
    missing_gtn = sum(1 for r in missing_runs if "_gtn_" in r)
    out(f"  Missing NTN runs: {missing_ntn}")
    out(f"  Missing GTN runs: {missing_gtn}")
    out()

    # Jobs to run by size
    out("=" * 80)
    out("JOBS TO RUN (by dataset size)")
    out("=" * 80)

    # Group incomplete experiments by size and trainer
    jobs_by_size_trainer = {size: {"ntn": [], "gtn": []} for size in SIZES}

    for (model, dataset, trainer), combo_s in stats["by_combo"].items():
        if combo_s["done"] < combo_s["total"]:
            size = DATASET_SIZES.get(dataset, "unknown")
            if size in jobs_by_size_trainer:
                jobs_by_size_trainer[size][trainer].append(
                    {
                        "model": model,
                        "dataset": dataset,
                        "done": combo_s["done"],
                        "total": combo_s["total"],
                        "missing": combo_s["total"] - combo_s["done"],
                    }
                )

    for size in SIZES:
        ntn_jobs = jobs_by_size_trainer[size]["ntn"]
        gtn_jobs = jobs_by_size_trainer[size]["gtn"]

        if not ntn_jobs and not gtn_jobs:
            out(f"\n{size.upper()}: All complete!")
            continue

        out(f"\n{size.upper()}:")

        if ntn_jobs:
            ntn_missing = sum(j["missing"] for j in ntn_jobs)
            out(f"  NTN ({len(ntn_jobs)} experiments, {ntn_missing} runs missing):")
            # Group by model for cleaner display
            by_model = defaultdict(list)
            for job in ntn_jobs:
                by_model[job["model"]].append(job["dataset"])
            for model, datasets in sorted(by_model.items()):
                out(f"    {model}: {', '.join(sorted(datasets))}")

        if gtn_jobs:
            gtn_missing = sum(j["missing"] for j in gtn_jobs)
            out(f"  GTN ({len(gtn_jobs)} experiments, {gtn_missing} runs missing):")
            by_model = defaultdict(list)
            for job in gtn_jobs:
                by_model[job["model"]].append(job["dataset"])
            for model, datasets in sorted(by_model.items()):
                out(f"    {model}: {', '.join(sorted(datasets))}")

    out()
    out("Suggested commands:")
    for size in SIZES:
        ntn_jobs = jobs_by_size_trainer[size]["ntn"]
        gtn_jobs = jobs_by_size_trainer[size]["gtn"]
        if ntn_jobs:
            out(
                f"  cd submit_ntn && bash submit_{size}_slurm.sh  # or submit_{size}_hpc.sh"
            )
        if gtn_jobs:
            out(
                f"  cd submit_gtn && bash submit_{size}_slurm.sh  # or submit_{size}_hpc.sh"
            )
    out()

    return "\n".join(lines)


def generate_markdown_report(data: dict, verbose: bool = False) -> str:
    """Generate the progress report as proper markdown."""
    lines = []

    def out(text: str = ""):
        lines.append(text)

    stats = data["stats"]
    missing_runs = data["missing_runs"]
    failed_runs = data["failed_runs"]
    singular_runs = data["singular_runs"]
    oom_runs = data.get("oom_runs", [])
    totals = data["totals"]

    total_expected = totals["expected"]
    total_completed = totals["completed"]
    total_success = totals["success"]
    total_singular = totals["singular"]
    total_oom = totals.get("oom", 0)
    total_failed = totals["failed"]

    # Header
    out("# GTN/NTN Ablation Study Progress")
    out()
    out("*Auto-generated by `check_progress.py`*")
    out()

    # Overall progress
    out("## Overall Progress")
    out()
    pct = total_completed / total_expected * 100 if total_expected > 0 else 0
    out(f"**Total:** {total_completed:,} / {total_expected:,} ({pct:.1f}%)")
    out()
    out(f"| Status | Count |")
    out("|--------|------:|")
    out(f"| Success | {total_success:,} |")
    out(f"| Singular | {total_singular:,} |")
    out(f"| OOM | {total_oom:,} |")
    out(f"| Failed | {total_failed:,} |")
    out(f"| Missing | {total_expected - total_completed:,} |")
    out()

    # By trainer
    out("## By Trainer")
    out()
    out("| Trainer | Done | Total | % | Success | Singular | OOM | Failed |")
    out("|---------|-----:|------:|--:|--------:|---------:|----:|-------:|")
    for trainer in TRAINERS:
        s = stats["by_trainer"][trainer]
        pct = s["done"] / s["total"] * 100 if s["total"] > 0 else 0
        out(
            f"| {trainer.upper()} | {s['done']:,} | {s['total']:,} | {pct:.1f}% | {s['success']:,} | {s['singular']:,} | {s['oom']:,} | {s['failed']:,} |"
        )
    out()

    # Model x Trainer matrix
    out("## Model x Trainer Matrix")
    out()
    out("| Model | NTN | GTN |")
    out("|-------|-----|-----|")
    for model in MODELS:
        ntn_s = stats["by_model_trainer"][(model, "ntn")]
        gtn_s = stats["by_model_trainer"][(model, "gtn")]
        ntn_pct = ntn_s["done"] / ntn_s["total"] * 100 if ntn_s["total"] > 0 else 0
        gtn_pct = gtn_s["done"] / gtn_s["total"] * 100 if gtn_s["total"] > 0 else 0
        ntn_str = f"{ntn_s['done']}/{ntn_s['total']} ({ntn_pct:.1f}%)"
        gtn_str = f"{gtn_s['done']}/{gtn_s['total']} ({gtn_pct:.1f}%)"
        out(f"| {model} | {ntn_str} | {gtn_str} |")
    out()

    # Size x Trainer matrix
    out("## By Dataset Size")
    out()
    out("| Size | NTN | GTN | Missing |")
    out("|------|-----|-----|--------:|")
    for size in SIZES:
        ntn_s = stats["by_size_trainer"][(size, "ntn")]
        gtn_s = stats["by_size_trainer"][(size, "gtn")]
        ntn_missing = ntn_s["total"] - ntn_s["done"]
        gtn_missing = gtn_s["total"] - gtn_s["done"]
        total_missing = ntn_missing + gtn_missing
        ntn_pct = ntn_s["done"] / ntn_s["total"] * 100 if ntn_s["total"] > 0 else 0
        gtn_pct = gtn_s["done"] / gtn_s["total"] * 100 if gtn_s["total"] > 0 else 0
        ntn_str = f"{ntn_s['done']}/{ntn_s['total']} ({ntn_pct:.1f}%)"
        gtn_str = f"{gtn_s['done']}/{gtn_s['total']} ({gtn_pct:.1f}%)"
        out(f"| {size.capitalize()} | {ntn_str} | {gtn_str} | {total_missing:,} |")
    out()

    # By dataset
    out("## By Dataset")
    out()
    complete_datasets = []
    incomplete_datasets = []
    for dataset in DATASETS:
        s = stats["by_dataset"][dataset]
        if s["done"] == s["total"]:
            complete_datasets.append(dataset)
        else:
            incomplete_datasets.append((dataset, s))

    if complete_datasets:
        out(f"**Complete:** {', '.join(complete_datasets)}")
        out()

    if incomplete_datasets:
        out("| Dataset | Done | Total | % |")
        out("|---------|-----:|------:|--:|")
        for dataset, s in incomplete_datasets:
            pct = s["done"] / s["total"] * 100 if s["total"] > 0 else 0
            out(f"| {dataset} | {s['done']:,} | {s['total']:,} | {pct:.1f}% |")
        out()

    # Incomplete experiments
    incomplete = [
        (k, v) for k, v in stats["by_combo"].items() if v["done"] < v["total"]
    ]
    if incomplete:
        out("## Incomplete Experiments")
        out()
        incomplete.sort(
            key=lambda x: x[1]["done"] / max(x[1]["total"], 1), reverse=True
        )
        show_count = len(incomplete) if verbose else 20
        out("| Model | Dataset | Trainer | Progress |")
        out("|-------|---------|---------|----------|")
        for (model, dataset, trainer), s in incomplete[:show_count]:
            pct = s["done"] / s["total"] * 100 if s["total"] > 0 else 0
            out(
                f"| {model} | {dataset} | {trainer.upper()} | {s['done']}/{s['total']} ({pct:.1f}%) |"
            )
        if not verbose and len(incomplete) > 20:
            out()
            out(f"*... and {len(incomplete) - 20} more*")
        out()

    # Summary of issues
    if singular_runs:
        out(f"## Singular Matrix Failures ({len(singular_runs)})")
        out()
        singular_by_exp = defaultdict(int)
        for run in singular_runs:
            exp = run.split("/")[0]
            singular_by_exp[exp] += 1
        out("| Experiment | Count |")
        out("|------------|------:|")
        for exp, count in sorted(singular_by_exp.items(), key=lambda x: -x[1])[:10]:
            out(f"| {exp} | {count} |")
        out()

    if failed_runs:
        out(f"## Other Failures ({len(failed_runs)})")
        out()
        for run in failed_runs[:10]:
            out(f"- `{run}`")
        if len(failed_runs) > 10:
            out(f"- *... and {len(failed_runs) - 10} more*")
        out()

    # Remaining work summary
    out("## Remaining Work Summary")
    out()
    remaining = total_expected - total_completed
    missing_ntn = sum(1 for r in missing_runs if "_ntn_" in r)
    missing_gtn = sum(1 for r in missing_runs if "_gtn_" in r)
    out(f"- **Runs remaining:** {remaining:,}")
    out(f"- **Experiments with missing runs:** {len(incomplete)}")
    out(f"- **Missing NTN runs:** {missing_ntn:,}")
    out(f"- **Missing GTN runs:** {missing_gtn:,}")
    out()

    # Jobs to run by size
    out("## Jobs to Run")
    out()

    jobs_by_size_trainer = {size: {"ntn": [], "gtn": []} for size in SIZES}
    for (model, dataset, trainer), combo_s in stats["by_combo"].items():
        if combo_s["done"] < combo_s["total"]:
            size = DATASET_SIZES.get(dataset, "unknown")
            if size in jobs_by_size_trainer:
                jobs_by_size_trainer[size][trainer].append(
                    {
                        "model": model,
                        "dataset": dataset,
                        "done": combo_s["done"],
                        "total": combo_s["total"],
                        "missing": combo_s["total"] - combo_s["done"],
                    }
                )

    for size in SIZES:
        ntn_jobs = jobs_by_size_trainer[size]["ntn"]
        gtn_jobs = jobs_by_size_trainer[size]["gtn"]

        if not ntn_jobs and not gtn_jobs:
            out(f"### {size.capitalize()}")
            out()
            out("All complete!")
            out()
            continue

        out(f"### {size.capitalize()}")
        out()

        if ntn_jobs:
            ntn_missing = sum(j["missing"] for j in ntn_jobs)
            out(f"**NTN** ({len(ntn_jobs)} experiments, {ntn_missing:,} runs missing):")
            out()
            by_model = defaultdict(list)
            for job in ntn_jobs:
                by_model[job["model"]].append(job["dataset"])
            for model, datasets in sorted(by_model.items()):
                out(f"- {model}: {', '.join(sorted(datasets))}")
            out()

        if gtn_jobs:
            gtn_missing = sum(j["missing"] for j in gtn_jobs)
            out(f"**GTN** ({len(gtn_jobs)} experiments, {gtn_missing:,} runs missing):")
            out()
            by_model = defaultdict(list)
            for job in gtn_jobs:
                by_model[job["model"]].append(job["dataset"])
            for model, datasets in sorted(by_model.items()):
                out(f"- {model}: {', '.join(sorted(datasets))}")
            out()

    # Suggested commands
    out("## Suggested Commands")
    out()
    out("```bash")
    for size in SIZES:
        ntn_jobs = jobs_by_size_trainer[size]["ntn"]
        gtn_jobs = jobs_by_size_trainer[size]["gtn"]
        if ntn_jobs:
            out(
                f"cd submit_ntn && bash submit_{size}_slurm.sh  # or submit_{size}_hpc.sh"
            )
        if gtn_jobs:
            out(
                f"cd submit_gtn && bash submit_{size}_slurm.sh  # or submit_{size}_hpc.sh"
            )
    out("```")
    out()

    return "\n".join(lines)


def write_readme(data: dict, verbose: bool = False) -> None:
    """Write the progress report to README.md as proper markdown."""
    readme_path = Path("README.md")
    content = generate_markdown_report(data, verbose=verbose)
    readme_path.write_text(content)


def main():
    parser = argparse.ArgumentParser(description="Check ablation study progress")
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Show all incomplete experiments"
    )
    parser.add_argument(
        "--missing", "-m", action="store_true", help="List all missing run paths"
    )
    parser.add_argument("--json", "-j", action="store_true", help="Output as JSON")
    parser.add_argument(
        "--no-readme", action="store_true", help="Don't write README.md file"
    )
    args = parser.parse_args()

    data = collect_stats()
    stats = data["stats"]
    missing_runs = data["missing_runs"]
    totals = data["totals"]

    if args.json:
        # JSON output mode
        output = {
            "totals": totals,
            "by_model": dict(stats["by_model"]),
            "by_dataset": dict(stats["by_dataset"]),
            "by_trainer": dict(stats["by_trainer"]),
            "incomplete_count": len(
                [c for c in stats["by_combo"].values() if c["done"] < c["total"]]
            ),
        }
        print(json.dumps(output, indent=2))
        return

    if args.missing:
        # Just print missing runs
        try:
            for run in sorted(missing_runs):
                print(run)
        except BrokenPipeError:
            pass  # Handle piping to head/less gracefully
        return

    # Generate and print report
    report = generate_report(data, verbose=args.verbose)
    print(report)

    # Write README.md unless disabled
    if not args.no_readme:
        write_readme(data, verbose=args.verbose)
        print("README.md updated.")


if __name__ == "__main__":
    main()
