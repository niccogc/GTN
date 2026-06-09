#!/usr/bin/env python3
"""
Test experiment progress tracker for GTN experiments.

Checks completed test runs in test_outputs and displays progress bars and summary statistics.
Test experiments evaluate best configurations on the test set with 10 different seeds.

Usage:
    python check_test_progress.py              # Default view
    python check_test_progress.py --verbose    # Show all incomplete experiments
    python check_test_progress.py --missing    # List all missing run paths
    python check_test_progress.py --json       # Output as JSON
    python check_test_progress.py --bash       # Generate bash arrays for missing experiments
"""

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

MODELS_NTN = [
    "CPDA",
    "CPDATypeI",
    "LMPO2",
    "LMPO2TypeI",
    "MPO2",
    "MPO2TypeI",
    "MMPO2",
    "MMPO2TypeI",
    "TNML_F",
    "TNML_P",
]

MODELS_GTN = [
    "BosonMPS",
    "CPDA",
    "CPDATypeI",
    "LMPO2",
    "LMPO2TypeI",
    "MPO2",
    "MPO2TypeI",
    "MMPO2",
    "MMPO2TypeI",
    "TNML_F",
    "TNML_P",
]

TRAINERS = ["ntn", "gtn"]
SIZES = ["small", "medium", "large"]

TEST_SEEDS = [
    836578142,
    895435625,
    2631647123,
    2487125586,
    3323088614,
    3313309148,
    3300558450,
    4053540165,
    2318036890,
    4234260150,
]

CONF_DIR = Path("conf")
TEST_OUTPUTS_DIR = Path("test_outputs")
TRACKING_FILE = Path("runs_tracking.csv")

# Models that use per-dataset best trainer from conf/best_conf/tnml/
TNML_MODELS = ["TNML_F", "TNML_P"]


def load_tnml_best_trainers() -> dict[str, dict[str, str]]:
    """Load best trainer per dataset for TNML models from conf/best_conf/tnml/*.yaml.
    
    Returns:
        Dict mapping model name -> {dataset: trainer}
    """
    best_trainers: dict[str, dict[str, str]] = {}
    tnml_dir = CONF_DIR / "best_conf" / "tnml"
    
    if not tnml_dir.exists():
        return best_trainers
    
    for yaml_file in tnml_dir.glob("*.yaml"):
        if yaml_file.stem.startswith("_"):
            continue
        
        model_name = yaml_file.stem.upper().replace("_", "_")  # tnml_f -> TNML_F
        # Map filename to model name
        model_map = {
            "tnml_f": "TNML_F",
            "tnml_p": "TNML_P",
        }
        model = model_map.get(yaml_file.stem, yaml_file.stem.upper())
        
        try:
            content = yaml_file.read_text()
            best_trainers[model] = {}
            
            # Parse YAML manually - look for dataset entries with trainer field
            current_dataset = None
            for line in content.split("\n"):
                # Match dataset name (indented key ending with :)
                dataset_match = re.match(r"^\s{2}(\w+):\s*$", line)
                if dataset_match:
                    current_dataset = dataset_match.group(1)
                    continue
                
                # Match trainer field
                trainer_match = re.match(r"^\s+trainer:\s*(\w+)", line)
                if trainer_match and current_dataset:
                    best_trainers[model][current_dataset] = trainer_match.group(1)
        except Exception:
            pass
    
    return best_trainers


# Load TNML best trainers at module load time
TNML_BEST_TRAINERS = load_tnml_best_trainers()


def load_tracking_file() -> pd.DataFrame:
    """Load the tracking CSV file (filtered to test_outputs only)."""
    if not TRACKING_FILE.exists():
        return pd.DataFrame()
    
    try:
        df = pd.read_csv(TRACKING_FILE)
        # Filter to test_outputs only
        df = df[df["output_path"].str.startswith("test_outputs", na=False)]
        return df
    except Exception as e:
        print(f"Warning: Could not load tracking file: {e}", file=sys.stderr)
        return pd.DataFrame()


def load_best_configs() -> dict[str, dict[str, dict]]:
    """Load best L/bond_dim configs for each trainer/model/dataset.
    
    Returns:
        Dict[trainer][model][dataset] -> {"L": int, "bond_dim": int}
    """
    best_configs: dict[str, dict[str, dict]] = {}
    best_conf_dir = CONF_DIR / "best_conf"
    
    for trainer_dir in best_conf_dir.iterdir():
        if not trainer_dir.is_dir():
            continue
        trainer = trainer_dir.name
        best_configs[trainer] = {}
        
        for yaml_file in trainer_dir.glob("*.yaml"):
            if yaml_file.stem.startswith("_"):
                continue
            
            # Map filename to model name
            model_map = {
                "tnml_f": "TNML_F",
                "tnml_p": "TNML_P",
                "mpo2": "MPO2",
                "lmpo2": "LMPO2",
                "mmpo2": "MMPO2",
                "cpda": "CPDA",
                "mpo2typei": "MPO2TypeI",
                "lmpo2typei": "LMPO2TypeI",
                "mmpo2typei": "MMPO2TypeI",
                "cpdatypei": "CPDATypeI",
                "bosonmps": "BosonMPS",
            }
            model = model_map.get(yaml_file.stem, yaml_file.stem.upper())
            
            try:
                content = yaml_file.read_text()
                best_configs[trainer][model] = {}
                
                current_dataset = None
                current_config: dict = {}
                
                for line in content.split("\n"):
                    # Match dataset name
                    dataset_match = re.match(r"^\s{2}(\w+):\s*$", line)
                    if dataset_match:
                        if current_dataset and current_config:
                            best_configs[trainer][model][current_dataset] = current_config.copy()
                        current_dataset = dataset_match.group(1)
                        current_config = {}
                        continue
                    
                    # Match L
                    l_match = re.match(r"^\s+L:\s*(\d+)", line)
                    if l_match and current_dataset:
                        current_config["L"] = int(l_match.group(1))
                    
                    # Match bond_dim
                    bd_match = re.match(r"^\s+bond_dim:\s*(\d+)", line)
                    if bd_match and current_dataset:
                        current_config["bond_dim"] = int(bd_match.group(1))
                
                # Don't forget last dataset
                if current_dataset and current_config:
                    best_configs[trainer][model][current_dataset] = current_config.copy()
                    
            except Exception:
                pass
    
    return best_configs


# Load best configs at module load time
BEST_CONFIGS = load_best_configs()


def get_test_ridge(trainer: str) -> float:
    """Get ridge value for test experiments."""
    # From conf/experiment/test_ntn.yaml and test_gtn.yaml
    if trainer == "ntn":
        return 5.0
    elif trainer == "gtn":
        return 0.005
    elif trainer == "dmrg":
        return 5.0
    return 5.0


def get_test_init_strength() -> float:
    """Get init_strength for test experiments."""
    return 0.1


def generate_test_run_id(trainer: str, model: str, dataset: str, seed: int) -> str:
    """Generate run_id for a test experiment run.
    
    Format matches what run.py generates via utils/tracking.py
    """
    # Get best config for this combo
    best = BEST_CONFIGS.get(trainer, {}).get(model, {}).get(dataset)
    if not best:
        return None
    
    L = best["L"]
    bond_dim = best["bond_dim"]
    ridge = get_test_ridge(trainer)
    init_strength = get_test_init_strength()
    
    run_id = (
        f"{trainer}_{dataset}_{model}"
        f"_L{L}_bd{bond_dim}"
        f"_rg{ridge}_init{init_strength}"
        f"_s{seed}"
    )
    
    return run_id


def load_datasets_from_conf() -> tuple[list[str], dict[str, str]]:
    """Load dataset names and their sizes from conf/dataset/*.yaml files."""
    datasets = []
    dataset_sizes = {}
    dataset_dir = CONF_DIR / "dataset"

    if not dataset_dir.exists():
        print(
            "Warning: conf/dataset directory not found, using defaults", file=sys.stderr
        )
        return _get_fallback_datasets()

    for yaml_file in sorted(dataset_dir.glob("*.yaml")):
        name = yaml_file.stem
        if name.startswith("_"):
            continue

        datasets.append(name)

        try:
            content = yaml_file.read_text()
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
    dataset_sizes = {d: "unknown" for d in datasets}
    return datasets, dataset_sizes


DATASETS, DATASET_SIZES = load_datasets_from_conf()


def get_models_for_trainer(trainer: str) -> list[str]:
    """Get models that can run with this trainer (excluding TNML which is per-dataset)."""
    if trainer == "ntn":
        return [m for m in MODELS_NTN if m not in TNML_MODELS]
    return [m for m in MODELS_GTN if m not in TNML_MODELS]


def should_run_tnml(model: str, dataset: str, trainer: str) -> bool:
    """Check if a TNML model should run on this dataset/trainer combo.
    
    TNML models only run on their best trainer per dataset.
    """
    if model not in TNML_MODELS:
        return True
    
    best_trainers = TNML_BEST_TRAINERS.get(model, {})
    best_trainer = best_trainers.get(dataset)
    
    # If no best config found, don't expect this run
    if best_trainer is None:
        return False
    
    return best_trainer == trainer


def get_experiment_path(trainer: str, model: str, dataset: str) -> Path:
    return TEST_OUTPUTS_DIR / trainer / model / dataset


def get_expected_subdirs() -> list[str]:
    return [f"seed_{seed}" for seed in TEST_SEEDS]


def check_result_from_tracking(tracking_df: pd.DataFrame, trainer: str, model: str, dataset: str, seed: int) -> dict:
    """Check if a run is completed based on tracking file.
    
    Matches by output_path pattern: test_outputs/{trainer}/{model}/{dataset}/seed_{seed}
    """
    if tracking_df.empty:
        return {"status": "missing", "success": None, "singular": None, "oom_error": None}
    
    # Match by output_path pattern
    expected_path = f"test_outputs/{trainer}/{model}/{dataset}/seed_{seed}"
    matches = tracking_df[tracking_df["output_path"] == expected_path]
    
    if matches.empty:
        return {"status": "missing", "success": None, "singular": None, "oom_error": None}
    
    row = matches.iloc[0]
    return {
        "status": "completed",
        "success": bool(row.get("success", False)),
        "singular": bool(row.get("singular", False)),
        "oom_error": bool(row.get("oom_error", False)),
        "val_quality": row.get("val_quality"),
    }


def check_result(result_path: Path, include_full_data: bool = False) -> dict:
    """Check a results.json file and return status info (filesystem fallback)."""
    if not result_path.exists():
        return {"status": "missing", "success": None, "singular": None, "oom_error": None}

    try:
        with open(result_path) as f:
            data = json.load(f)
        result = {
            "status": "completed",
            "success": data.get("success", False),
            "singular": data.get("singular", False),
            "oom_error": data.get("oom_error", False),
            "test_quality": data.get("test_quality"),
            "val_quality": data.get("val_quality"),
        }
        if include_full_data:
            result["config"] = data.get("config", {})
            result["gpu_memory"] = data.get("gpu_memory", {})
            result["dataset_info"] = data.get("dataset_info", {})
        return result
    except (json.JSONDecodeError, KeyError):
        return {"status": "corrupted", "success": None, "singular": None, "oom_error": None}


def print_progress_bar(completed: int, total: int, width: int = 40, prefix: str = "") -> str:
    if total == 0:
        pct = 0
        filled = 0
    else:
        pct = completed / total * 100
        filled = int(width * completed / total)

    bar = "█" * filled + "░" * (width - filled)
    return f"{prefix}[{bar}] {completed:4d}/{total:4d} ({pct:5.1f}%)"


def collect_stats():
    """Collect all statistics from tracking file."""
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

    expected_seeds = len(TEST_SEEDS)
    
    # Load tracking file
    tracking_df = load_tracking_file()
    if tracking_df.empty:
        print("Warning: No tracking data found. Run: python scripts/backfill_tracking.py --outputs-dir test_outputs --append", file=sys.stderr)

    # Iterate through all expected combinations
    for trainer in TRAINERS:
        # Regular models (non-TNML)
        models = get_models_for_trainer(trainer)
        
        # Add TNML models - they will be filtered per-dataset
        all_models = models + TNML_MODELS
        
        for model in all_models:
            for dataset in DATASETS:
                # Skip TNML models that shouldn't run on this trainer for this dataset
                if model in TNML_MODELS and not should_run_tnml(model, dataset, trainer):
                    continue
                
                # Skip if no best config exists for this combo
                if not BEST_CONFIGS.get(trainer, {}).get(model, {}).get(dataset):
                    continue
                exp_path_str = f"{trainer}/{model}/{dataset}"

                combo_key = (model, dataset, trainer)
                model_trainer_key = (model, trainer)
                combo_stats = {
                    "done": 0,
                    "total": expected_seeds,
                    "success": 0,
                    "failed": 0,
                    "singular": 0,
                    "oom": 0,
                    "missing": [],
                }

                for seed in TEST_SEEDS:
                    subdir = f"seed_{seed}"
                    # Check tracking file by output_path pattern
                    result = check_result_from_tracking(tracking_df, trainer, model, dataset, seed)

                    total_expected += 1
                    stats["by_model"][model]["total"] += 1
                    stats["by_dataset"][dataset]["total"] += 1
                    stats["by_trainer"][trainer]["total"] += 1
                    stats["by_model_trainer"][model_trainer_key]["total"] += 1

                    if result["status"] == "completed":
                        if result.get("oom_error"):
                            # OOM - not actually done, treat as needing re-run
                            total_oom += 1
                            combo_stats["oom"] += 1
                            stats["by_model"][model]["oom"] += 1
                            stats["by_dataset"][dataset]["oom"] += 1
                            stats["by_trainer"][trainer]["oom"] += 1
                            stats["by_model_trainer"][model_trainer_key]["oom"] += 1
                            oom_runs.append(f"{exp_path_str}/{subdir}")
                        else:
                            # Actually completed (success, singular, or non-OOM failure)
                            total_completed += 1
                            combo_stats["done"] += 1
                            stats["by_model"][model]["done"] += 1
                            stats["by_dataset"][dataset]["done"] += 1
                            stats["by_trainer"][trainer]["done"] += 1
                            stats["by_model_trainer"][model_trainer_key]["done"] += 1

                            if result["success"]:
                                total_success += 1
                                combo_stats["success"] += 1
                                stats["by_model"][model]["success"] += 1
                                stats["by_dataset"][dataset]["success"] += 1
                                stats["by_trainer"][trainer]["success"] += 1
                                stats["by_model_trainer"][model_trainer_key]["success"] += 1
                            elif result["singular"]:
                                total_singular += 1
                                combo_stats["singular"] += 1
                                stats["by_model"][model]["singular"] += 1
                                stats["by_dataset"][dataset]["singular"] += 1
                                stats["by_trainer"][trainer]["singular"] += 1
                                stats["by_model_trainer"][model_trainer_key]["singular"] += 1
                                singular_runs.append(f"{exp_path_str}/{subdir}")
                            else:
                                total_failed += 1
                                combo_stats["failed"] += 1
                                stats["by_model"][model]["failed"] += 1
                                stats["by_dataset"][dataset]["failed"] += 1
                                stats["by_trainer"][trainer]["failed"] += 1
                                stats["by_model_trainer"][model_trainer_key]["failed"] += 1
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
    out("                     GTN/NTN TEST EXPERIMENT PROGRESS")
    out("=" * 80)
    out()
    out(f"Test experiments evaluate best configs on test set with {len(TEST_SEEDS)} seeds")
    out(f"Datasets: {len(DATASETS)} | Models (NTN): {len(MODELS_NTN)} | Models (GTN): {len(MODELS_GTN)}")
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
    
    all_models = sorted(set(MODELS_NTN + MODELS_GTN))
    for model in all_models:
        ntn_s = stats["by_model_trainer"][(model, "ntn")]
        gtn_s = stats["by_model_trainer"][(model, "gtn")]
        ntn_pct = ntn_s["done"] / ntn_s["total"] * 100 if ntn_s["total"] > 0 else 0
        gtn_pct = gtn_s["done"] / gtn_s["total"] * 100 if gtn_s["total"] > 0 else 0
        
        if ntn_s["total"] > 0:
            ntn_str = f"{ntn_s['done']:4d}/{ntn_s['total']:4d} ({ntn_pct:5.1f}%)"
        else:
            ntn_str = "      N/A       "
            
        if gtn_s["total"] > 0:
            gtn_str = f"{gtn_s['done']:4d}/{gtn_s['total']:4d} ({gtn_pct:5.1f}%)"
        else:
            gtn_str = "      N/A       "
            
        out(f"{model:<12s} | {ntn_str:^20s} | {gtn_str:^20s}")
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
        # Group by model
        singular_by_model = defaultdict(int)
        for run in singular_runs:
            parts = run.split("/")
            model = parts[1] if len(parts) > 1 else "unknown"
            singular_by_model[model] += 1
        for model, count in sorted(singular_by_model.items(), key=lambda x: -x[1])[:10]:
            out(f"  {model}: {count}")
        out()

    if oom_runs:
        out(f"CUDA OUT OF MEMORY FAILURES: {len(oom_runs)}")
        out("-" * 80)
        # Group by model
        oom_by_model = defaultdict(int)
        for run in oom_runs:
            parts = run.split("/")
            model = parts[1] if len(parts) > 1 else "unknown"
            oom_by_model[model] += 1
        for model, count in sorted(oom_by_model.items(), key=lambda x: -x[1])[:10]:
            out(f"  {model}: {count}")
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
    missing_ntn = sum(1 for r in missing_runs if r.startswith("ntn/"))
    missing_gtn = sum(1 for r in missing_runs if r.startswith("gtn/"))
    out(f"  Missing NTN runs: {missing_ntn}")
    out(f"  Missing GTN runs: {missing_gtn}")
    out()

    # Jobs to run by trainer
    out("=" * 80)
    out("JOBS TO RUN")
    out("=" * 80)

    jobs_by_trainer = {"ntn": [], "gtn": []}

    for (model, dataset, trainer), combo_s in stats["by_combo"].items():
        if combo_s["done"] < combo_s["total"]:
            jobs_by_trainer[trainer].append(
                {
                    "model": model,
                    "dataset": dataset,
                    "done": combo_s["done"],
                    "total": combo_s["total"],
                    "missing": combo_s["total"] - combo_s["done"],
                }
            )

    for trainer in TRAINERS:
        jobs = jobs_by_trainer[trainer]
        if not jobs:
            out(f"\n{trainer.upper()}: All complete!")
            continue

        total_missing = sum(j["missing"] for j in jobs)
        out(f"\n{trainer.upper()} ({len(jobs)} experiments, {total_missing} runs missing):")

        # Group by model for cleaner display
        by_model = defaultdict(list)
        for job in jobs:
            by_model[job["model"]].append(job["dataset"])
        for model, datasets in sorted(by_model.items()):
            out(f"  {model}: {', '.join(sorted(datasets))}")

    out()
    out("Suggested commands:")
    out("  cd submit_test && bash submit_cpu_ntn.sh  # For NTN test runs")
    out("  cd submit_test && bash submit_cpu_gtn.sh  # For GTN test runs")
    out()

    return "\n".join(lines)


def generate_bash_arrays(data: dict) -> str:
    """Generate bash declare -a arrays for missing test experiments."""
    MODEL_TO_VAR = {
        "MPO2": "mpo2",
        "LMPO2": "lmpo2",
        "MMPO2": "mmpo2",
        "MPO2TypeI": "mpo2_typei",
        "LMPO2TypeI": "lmpo2_typei",
        "MMPO2TypeI": "mmpo2_typei",
        "CPDA": "cpda",
        "CPDATypeI": "cpda_typei",
        "TNML_P": "tnml_p",
        "TNML_F": "tnml_f",
        "BosonMPS": "bosonmps",
    }
    stats = data["stats"]
    lines = []

    for trainer in TRAINERS:
        trainer_upper = trainer.upper()
        var_name = f"COMBINATIONS_{trainer_upper}"

        missing_combos = []
        for (model, dataset, combo_trainer), combo_s in stats["by_combo"].items():
            if combo_trainer == trainer and combo_s["done"] < combo_s["total"]:
                missing_combos.append((model, dataset))

        missing_combos.sort(key=lambda x: (x[0], x[1]))
        i = 0
        by_model = defaultdict(list)
        for model, dataset in missing_combos:
            by_model[model].append(dataset)
            i += 1

        lines.append(f"# {i} Experiments")
        lines.append(f"declare -a {var_name}=(")

        for model in sorted(by_model.keys()):
            datasets = sorted(by_model[model])
            model_var = MODEL_TO_VAR.get(model, model.lower())
            lines.append(f"    # {model} ({len(datasets)} datasets)")
            for dataset in datasets:
                lines.append(f'    "{model_var} {dataset}"')

        lines.append(")")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def main():
    parser = argparse.ArgumentParser(description="Check test experiment progress")
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Show all incomplete experiments"
    )
    parser.add_argument(
        "--missing", "-m", action="store_true", help="List all missing run paths"
    )
    parser.add_argument("--json", "-j", action="store_true", help="Output as JSON")
    parser.add_argument(
        "--bash",
        action="store_true",
        help="Print missing experiments as bash declare -a COMBINATIONS arrays",
    )
    args = parser.parse_args()

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

    if args.bash:
        bash_content = generate_bash_arrays(data)
        print(bash_content)
        # Also save to missing_test.env
        Path("missing_test.env").write_text(bash_content)
        print("missing_test.env updated.", file=sys.stderr)
        return

    report = generate_report(data, verbose=args.verbose)
    print(report)


if __name__ == "__main__":
    main()
