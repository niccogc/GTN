#!/usr/bin/env python3
"""
Ablation study progress tracker for GTN experiments.

Checks completed runs and displays progress bars and summary statistics.

Usage:
    python check_progress.py              # Default view
    python check_progress.py --verbose    # Show all incomplete experiments
    python check_progress.py --missing    # List all missing run paths
    python check_progress.py --json       # Output as JSON
    python check_progress.py --oom        # Deep OOM analysis with parameter breakdowns
    python check_progress.py --oom-detail # Show all OOM run paths
"""

import argparse
import json
import os
import re
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
    "TNML_P",
    "TNML_F",
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


def check_result(result_path: Path, include_full_data: bool = False) -> dict:
    """Check a results.json file and return status info.

    Args:
        result_path: Path to results.json
        include_full_data: If True, include full config and gpu_memory data
    """
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
        result = {
            "status": "completed",
            "success": data.get("success", False),
            "singular": data.get("singular", False),
            "oom_error": data.get("oom_error", False),
            "val_quality": data.get("val_quality"),
        }
        if include_full_data:
            result["config"] = data.get("config", {})
            result["gpu_memory"] = data.get("gpu_memory", {})
            result["dataset_info"] = data.get("dataset_info", {})
        return result
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


def collect_oom_details() -> list[dict]:
    """Collect detailed information about all OOM runs by reading results.json files."""
    oom_details = []

    for model in MODELS:
        expected_subdirs = get_expected_subdirs(model)

        for dataset in DATASETS:
            for trainer in TRAINERS:
                exp_dir = get_experiment_path(model, dataset, trainer)

                for subdir in expected_subdirs:
                    result_path = exp_dir / subdir / "results.json"
                    result = check_result(result_path, include_full_data=True)

                    if result.get("oom_error"):
                        params = parse_subdir_params(subdir)
                        detail = {
                            "path": str(exp_dir / subdir),
                            "model": model,
                            "dataset": dataset,
                            "trainer": trainer,
                            "size": DATASET_SIZES.get(dataset, "unknown"),
                            **params,
                        }

                        gpu_mem = result.get("gpu_memory", {})
                        if gpu_mem:
                            after = gpu_mem.get("after", {})
                            detail["peak_gb"] = after.get("peak_allocated_gb")
                            detail["total_gpu_gb"] = after.get("total_gb")
                            detail["device"] = after.get("device_name", "unknown")

                        ds_info = result.get("dataset_info", {})
                        if ds_info:
                            detail["n_features"] = ds_info.get("n_features")
                            detail["n_train"] = ds_info.get("n_train")

                        oom_details.append(detail)

    return oom_details


def parse_subdir_params(subdir: str) -> dict:
    """Parse L, bond_dim, seed, rf from subdir name like L4_bd12_seed47311_rf0.3_bondmpo1."""
    params = {}

    for pattern, key, conv in [
        (r"L(\d+)", "L", int),
        (r"bd(\d+)", "bond_dim", int),
        (r"seed(\d+)", "seed", int),
        (r"rf([\d.]+)", "reduction_factor", float),
    ]:
        m = re.search(pattern, subdir)
        if m:
            params[key] = conv(m.group(1))

    return params


def generate_oom_report(oom_details: list[dict], show_all_paths: bool = False) -> str:
    """Generate detailed OOM analysis report."""
    lines = []

    def out(text: str = ""):
        lines.append(text)

    if not oom_details:
        out("No OOM errors found.")
        return "\n".join(lines)

    out("=" * 100)
    out("                           CUDA OUT OF MEMORY ANALYSIS")
    out("=" * 100)
    out()
    out(f"Total OOM runs: {len(oom_details)}")
    out()

    # =========================================================================
    # BY MODEL
    # =========================================================================
    out("OOM BY MODEL")
    out("-" * 100)
    by_model = defaultdict(list)
    for d in oom_details:
        by_model[d["model"]].append(d)

    for model in MODELS:
        count = len(by_model.get(model, []))
        if count > 0:
            pct = count / len(oom_details) * 100
            out(f"  {model:<15s}: {count:4d} ({pct:5.1f}%)")
    out()

    # =========================================================================
    # BY DATASET SIZE
    # =========================================================================
    out("OOM BY DATASET SIZE")
    out("-" * 100)
    by_size = defaultdict(list)
    for d in oom_details:
        by_size[d["size"]].append(d)

    for size in SIZES:
        count = len(by_size.get(size, []))
        if count > 0:
            pct = count / len(oom_details) * 100
            out(f"  {size:<10s}: {count:4d} ({pct:5.1f}%)")
    out()

    # =========================================================================
    # BY BOND_DIM
    # =========================================================================
    out("OOM BY BOND DIMENSION")
    out("-" * 100)
    by_bd = defaultdict(list)
    for d in oom_details:
        bd = d.get("bond_dim", "unknown")
        by_bd[bd].append(d)

    for bd in sorted(by_bd.keys(), key=lambda x: (isinstance(x, str), x)):
        count = len(by_bd[bd])
        pct = count / len(oom_details) * 100
        out(f"  bond_dim={bd:<3}: {count:4d} ({pct:5.1f}%)")
    out()

    # =========================================================================
    # BY L (number of sites)
    # =========================================================================
    out("OOM BY NUMBER OF SITES (L)")
    out("-" * 100)
    by_L = defaultdict(list)
    for d in oom_details:
        L = d.get("L", "unknown")
        by_L[L].append(d)

    for L in sorted(by_L.keys(), key=lambda x: (isinstance(x, str), x)):
        count = len(by_L[L])
        pct = count / len(oom_details) * 100
        out(f"  L={L}: {count:4d} ({pct:5.1f}%)")
    out()

    # =========================================================================
    # BY DATASET
    # =========================================================================
    out("OOM BY DATASET")
    out("-" * 100)
    by_dataset = defaultdict(list)
    for d in oom_details:
        by_dataset[d["dataset"]].append(d)

    sorted_datasets = sorted(by_dataset.items(), key=lambda x: -len(x[1]))
    for dataset, items in sorted_datasets:
        size = DATASET_SIZES.get(dataset, "?")
        count = len(items)
        pct = count / len(oom_details) * 100
        out(f"  {dataset:<18s} ({size:<6s}): {count:4d} ({pct:5.1f}%)")
    out()

    # =========================================================================
    # CROSS-ANALYSIS: MODEL x BOND_DIM
    # =========================================================================
    out("OOM HEATMAP: MODEL x BOND_DIM")
    out("-" * 100)

    all_bds = sorted(
        set(d.get("bond_dim", 0) for d in oom_details if d.get("bond_dim"))
    )
    header = f"{'Model':<15s}"
    for bd in all_bds:
        header += f" | bd={bd:<3d}"
    header += " | Total"
    out(header)
    out("-" * len(header))

    for model in MODELS:
        model_ooms = by_model.get(model, [])
        if not model_ooms:
            continue

        row = f"{model:<15s}"
        for bd in all_bds:
            count = sum(1 for d in model_ooms if d.get("bond_dim") == bd)
            row += f" | {count:>6d}" if count > 0 else " |      -"
        row += f" | {len(model_ooms):>5d}"
        out(row)
    out()

    # =========================================================================
    # CROSS-ANALYSIS: DATASET x MODEL (top offenders)
    # =========================================================================
    out("TOP OOM COMBINATIONS (Dataset x Model)")
    out("-" * 100)

    combo_counts = defaultdict(int)
    for d in oom_details:
        key = (d["dataset"], d["model"])
        combo_counts[key] += 1

    sorted_combos = sorted(combo_counts.items(), key=lambda x: -x[1])[:20]
    for (dataset, model), count in sorted_combos:
        size = DATASET_SIZES.get(dataset, "?")
        out(f"  {dataset:<18s} + {model:<15s} ({size}): {count:4d}")
    out()

    # =========================================================================
    # PARAMETER CORRELATION ANALYSIS
    # =========================================================================
    out("OOM PARAMETER PATTERNS")
    out("-" * 100)

    l4_count = sum(1 for d in oom_details if d.get("L") == 4)
    l3_count = sum(1 for d in oom_details if d.get("L") == 3)
    out(f"  L=4 vs L=3: {l4_count} vs {l3_count}")

    high_bd = sum(1 for d in oom_details if d.get("bond_dim", 0) >= 12)
    low_bd = sum(1 for d in oom_details if d.get("bond_dim", 0) < 12)
    out(f"  bond_dim >= 12 vs < 12: {high_bd} vs {low_bd}")

    large_count = sum(1 for d in oom_details if d.get("size") == "large")
    other_count = len(oom_details) - large_count
    out(f"  Large datasets vs others: {large_count} vs {other_count}")

    lmpo2_ooms = [d for d in oom_details if "LMPO2" in d["model"]]
    if lmpo2_ooms:
        out()
        out("  LMPO2 reduction_factor breakdown:")
        by_rf = defaultdict(int)
        for d in lmpo2_ooms:
            rf = d.get("reduction_factor", "unknown")
            by_rf[rf] += 1
        for rf, count in sorted(
            by_rf.items(), key=lambda x: (isinstance(x[0], str), x[0])
        ):
            out(f"    rf={rf}: {count}")
    out()

    # =========================================================================
    # GPU MEMORY ANALYSIS
    # =========================================================================
    peak_gbs: list[float] = [
        d["peak_gb"]
        for d in oom_details
        if d.get("peak_gb") is not None and isinstance(d["peak_gb"], (int, float))
    ]
    if peak_gbs:
        out("GPU PEAK MEMORY AT OOM")
        out("-" * 100)
        out(f"  Samples with peak_gb data: {len(peak_gbs)}")
        out(f"  Min peak: {min(peak_gbs):.2f} GB")
        out(f"  Max peak: {max(peak_gbs):.2f} GB")
        out(f"  Avg peak: {sum(peak_gbs) / len(peak_gbs):.2f} GB")

        by_device = defaultdict(list)
        for d in oom_details:
            if d.get("peak_gb") is not None:
                by_device[d.get("device", "unknown")].append(d["peak_gb"])

        if len(by_device) > 1:
            out()
            out("  By GPU device:")
            for device, peaks in sorted(by_device.items()):
                out(
                    f"    {device}: avg={sum(peaks) / len(peaks):.2f} GB, n={len(peaks)}"
                )
        out()

    # =========================================================================
    # FEATURE COUNT ANALYSIS
    # =========================================================================
    n_features_list = [d.get("n_features") for d in oom_details if d.get("n_features")]
    if n_features_list:
        out("OOM BY FEATURE COUNT")
        out("-" * 100)

        by_features = defaultdict(int)
        for d in oom_details:
            nf = d.get("n_features")
            if nf:
                by_features[nf] += 1

        sorted_features = sorted(by_features.items(), key=lambda x: -x[1])[:10]
        for nf, count in sorted_features:
            pct = count / len(oom_details) * 100
            out(f"  n_features={nf:<3d}: {count:4d} ({pct:5.1f}%)")
        out()

    # =========================================================================
    # ACTIONABLE INSIGHTS
    # =========================================================================
    out("=" * 100)
    out("ACTIONABLE INSIGHTS")
    out("=" * 100)
    out()

    if sorted_combos:
        worst_combo = sorted_combos[0]
        out(
            f"1. WORST COMBO: {worst_combo[0][0]} + {worst_combo[0][1]} ({worst_combo[1]} OOMs)"
        )
        out(
            f"   → Consider reducing bond_dim or using gradient checkpointing for this combo"
        )
        out()

    if l4_count > l3_count * 2:
        out(f"2. L=4 DOMINATES: {l4_count} OOMs vs {l3_count} for L=3")
        out("   → Memory scales exponentially with L; consider L=3 for large datasets")
        out()

    if high_bd > low_bd * 2:
        out(f"3. HIGH BOND_DIM DOMINATES: {high_bd} OOMs for bd>=12")
        out("   → Reduce bond_dim for memory-constrained runs")
        out()

    if large_count > other_count:
        out(
            f"4. LARGE DATASETS DOMINATE: {large_count} OOMs ({large_count / len(oom_details) * 100:.1f}%)"
        )
        out("   → Large datasets need smaller models or more GPU memory")
        out()

    # =========================================================================
    # ALL OOM PATHS (if requested)
    # =========================================================================
    if show_all_paths:
        out()
        out("=" * 100)
        out("ALL OOM RUN PATHS")
        out("=" * 100)
        for d in sorted(
            oom_details, key=lambda x: (x["model"], x["dataset"], x.get("bond_dim", 0))
        ):
            bd = d.get("bond_dim", "?")
            L = d.get("L", "?")
            out(f"  {d['path']}  (L={L}, bd={bd})")
        out()

    return "\n".join(lines)


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
    parser.add_argument(
        "--oom", action="store_true", help="Deep OOM analysis with parameter breakdowns"
    )
    parser.add_argument(
        "--oom-detail",
        action="store_true",
        help="Show all OOM run paths (implies --oom)",
    )
    parser.add_argument(
        "--oom-json", action="store_true", help="Output OOM details as JSON"
    )
    args = parser.parse_args()

    if args.oom or args.oom_detail or args.oom_json:
        print("Scanning for OOM runs...", file=sys.stderr)
        oom_details = collect_oom_details()

        if args.oom_json:
            print(json.dumps(oom_details, indent=2))
            return

        report = generate_oom_report(oom_details, show_all_paths=args.oom_detail)
        print(report)
        return

    data = collect_stats()
    stats = data["stats"]
    missing_runs = data["missing_runs"]
    totals = data["totals"]

    if args.json:
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
        try:
            for run in sorted(missing_runs):
                print(run)
        except BrokenPipeError:
            pass
        return

    report = generate_report(data, verbose=args.verbose)
    print(report)

    if not args.no_readme:
        write_readme(data, verbose=args.verbose)
        print("README.md updated.")


if __name__ == "__main__":
    main()
