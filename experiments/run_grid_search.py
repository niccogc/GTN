# type: ignore
"""
Grid search experiment runner.
Reads JSON configuration and runs all parameter combinations with tracking.
"""

import os
import sys
import json
import argparse
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np

from experiments.config_parser import load_config, create_experiment_plan, print_experiment_summary
from experiments.dataset_loader import load_dataset
from experiments.trackers import create_tracker, TrackerError

from model.base.NTN import NTN
from model.base.NTN_Ensemble import NTN_Ensemble
from model.losses import MSELoss, CrossEntropyLoss
from model.utils import REGRESSION_METRICS, CLASSIFICATION_METRICS, compute_quality, create_inputs
from model.standard import MPO2, LMPO2, MMPO2
from model.typeI import MPO2TypeI, LMPO2TypeI, MMPO2TypeI
from model.exceptions import SingularMatrixError

from experiments.device_utils import DEVICE, move_tn_to_device, move_data_to_device

torch.set_default_dtype(torch.float64)


def get_result_filepath(output_dir: str, run_id: str) -> str:
    return os.path.join(output_dir, f"{run_id}.json")


def run_already_completed(output_dir: str, run_id: str) -> tuple[bool, bool, bool, str | None]:
    """
    Check if a run has already been attempted.

    Returns:
        (was_attempted, was_successful, is_singular, error_message)
        - is_singular: True if failed due to singular matrix (should skip permanently)
    """
    result_file = get_result_filepath(output_dir, run_id)

    if not os.path.exists(result_file):
        return False, False, False, None

    try:
        with open(result_file, "r") as f:
            result = json.load(f)
        success = result.get("success", False)
        singular = result.get("singular", False)
        error = result.get("error", None)
        return True, success, singular, error
    except:
        return False, False, False, None


def create_model(model_name: str, params: dict, input_dim: int, output_dim: int):
    """Create model instance based on model name and parameters."""

    if model_name == "MPO2":
        return MPO2(
            L=params["L"],
            bond_dim=params["bond_dim"],
            phys_dim=input_dim,
            output_dim=output_dim,
            output_site=params.get("output_site"),
            init_strength=params.get("init_strength", 0.1),
        )

    elif model_name == "LMPO2":
        if "reduced_dim" in params:
            reduced_dim = params["reduced_dim"]
        elif "reduction_factor" in params:
            reduced_dim = max(2, int(input_dim * params["reduction_factor"]))
        else:
            raise ValueError("LMPO2 requires either reduced_dim or reduction_factor")
        return LMPO2(
            L=params["L"],
            bond_dim=params["bond_dim"],
            phys_dim=input_dim,
            output_dim=output_dim,
            reduced_dim=reduced_dim,
            output_site=params.get("output_site"),
            bond_dim_mpo=params.get("bond_dim_mpo", 2),
        )

    elif model_name == "MMPO2":
        return MMPO2(
            L=params["L"],
            bond_dim=params["bond_dim"],
            phys_dim=input_dim,
            output_dim=output_dim,
            output_site=params.get("output_site"),
        )

    elif model_name == "MPO2TypeI":
        return MPO2TypeI(
            max_sites=params["L"],
            bond_dim=params["bond_dim"],
            phys_dim=input_dim,
            output_dim=output_dim,
            output_site=params.get("output_site"),
            init_strength=params.get("init_strength", 0.1),
        )

    elif model_name == "LMPO2TypeI":
        if "reduced_dim" in params:
            reduced_dim = params["reduced_dim"]
        elif "reduction_factor" in params:
            reduced_dim = max(2, int(input_dim * params["reduction_factor"]))
        else:
            raise ValueError("LMPO2TypeI requires either reduced_dim or reduction_factor")
        return LMPO2TypeI(
            max_sites=params["L"],
            bond_dim=params["bond_dim"],
            phys_dim=input_dim,
            reduced_dim=reduced_dim,
            output_dim=output_dim,
            output_site=params.get("output_site"),
            init_strength=params.get("init_strength", 0.1),
        )

    elif model_name == "MMPO2TypeI":
        return MMPO2TypeI(
            max_sites=params["L"],
            bond_dim=params["bond_dim"],
            phys_dim=input_dim,
            output_dim=output_dim,
            output_site=params.get("output_site"),
            init_strength=params.get("init_strength", 0.1),
        )

    else:
        raise ValueError(f"Unknown model: {model_name}")


def run_single_experiment(
    experiment: dict,
    data: dict,
    input_dim: int,
    output_dim: int,
    verbose: bool = False,
    tracker=None,
):
    """Run a single experiment with given parameters."""

    params = experiment["params"]
    seed = experiment["seed"]
    task = experiment["task"]

    torch.manual_seed(seed)
    np.random.seed(seed)

    data = move_data_to_device(data)

    if task == "regression":
        loss_fn = MSELoss()
        eval_metrics = REGRESSION_METRICS
    else:
        loss_fn = CrossEntropyLoss()
        eval_metrics = CLASSIFICATION_METRICS

    n_epochs = params.get("n_epochs", 10)
    jitter_start = params.get("jitter_start", 0.001)
    jitter_decay = params.get("jitter_decay", 0.95)
    jitter_min = params.get("jitter_min", 0.001)

    jitter_schedule = [
        max(jitter_start * (jitter_decay**epoch), jitter_min) for epoch in range(n_epochs)
    ]

    model_name = params["model"]
    model = create_model(model_name, params, input_dim, output_dim)
    is_typeI = model_name.endswith("TypeI")

    if is_typeI:
        for tn in model.tns:
            move_tn_to_device(tn)
    else:
        move_tn_to_device(model.tn)

    if is_typeI:
        not_trainable_tags = getattr(model, "not_trainable_tags", None)
        ntn = NTN_Ensemble(
            tns=model.tns,
            input_dims_list=model.input_dims_list,
            input_labels_list=model.input_labels_list,
            output_dims=model.output_dims,
            loss=loss_fn,
            X_train=data["X_train"],
            y_train=data["y_train"],
            X_val=data["X_val"],
            y_val=data["y_val"],
            batch_size=params.get("batch_size", 100),
            not_trainable_tags=not_trainable_tags,
        )
        loader_val = ntn.val_data
    else:
        loader_train = create_inputs(
            X=data["X_train"],
            y=data["y_train"],
            input_labels=model.input_labels,
            output_labels=model.output_dims,
            batch_size=params.get("batch_size", 100),
            append_bias=False,
        )

        loader_val = create_inputs(
            X=data["X_val"],
            y=data["y_val"],
            input_labels=model.input_labels,
            output_labels=model.output_dims,
            batch_size=params.get("batch_size", 100),
            append_bias=False,
        )

        ntn = NTN(
            tn=model.tn,
            output_dims=model.output_dims,
            input_dims=model.input_dims,
            loss=loss_fn,
            data_stream=loader_train,
        )

    if is_typeI:
        n_parameters = sum(t.data.numel() for tn in model.tns for t in tn.tensors)
    else:
        n_parameters = sum(t.data.numel() for t in model.tn.tensors)

    def callback_init(scores_train, scores_val, info):
        if tracker:
            hparams = {
                "seed": seed,
                "model": model_name,
                "dataset": experiment["dataset"],
                "n_parameters": n_parameters,
                **params,
            }
            tracker.log_hparams(hparams)

            metrics = {
                "train_loss": scores_train["loss"],
                "train_quality": compute_quality(scores_train),
                "val_loss": scores_val["loss"],
                "val_quality": compute_quality(scores_val),
            }
            tracker.log_metrics(metrics, step=-1)

    def callback_epoch(epoch, scores_train, scores_val, info):
        if tracker:
            metrics = {
                "train_loss": scores_train["loss"],
                "train_quality": compute_quality(scores_train),
                "val_loss": scores_val["loss"],
                "val_quality": compute_quality(scores_val),
                "reg_loss": info["reg_loss"],
                "jitter": info["jitter"],
            }

            if info["weight_norm_sq"] is not None:
                metrics["weight_norm_sq"] = info["weight_norm_sq"]

            tracker.log_metrics(metrics, step=epoch)

    try:
        scores_train, scores_val = ntn.fit(
            n_epochs=n_epochs,
            regularize=True,
            jitter=jitter_schedule,
            eval_metrics=eval_metrics,
            val_data=loader_val,
            verbose=verbose,
            callback_init=callback_init,
            callback_epoch=callback_epoch,
            adaptive_jitter=params.get("adaptive_jitter", True),
            patience=params.get("patience", 5),
            min_delta=params.get("min_delta", 0.001),
            train_selection=params.get("train_selection", False),
        )

        train_loss = scores_train["loss"]
        train_quality = compute_quality(scores_train)
        val_loss = scores_val["loss"]
        val_quality = compute_quality(scores_val)

        success = val_quality is not None and val_quality > 0

        result = {
            "run_id": experiment["run_id"],
            "seed": seed,
            "model": model_name,
            "grid_params": experiment["grid_params"],
            "n_parameters": n_parameters,
            "train_loss": float(train_loss),
            "train_quality": float(train_quality),
            "val_loss": float(val_loss),
            "val_quality": float(val_quality),
            "success": success,
            "singular": ntn.singular_encountered,
        }

        if tracker:
            summary = {**result, "n_parameters": n_parameters}
            tracker.log_summary(summary)

        return result

    except SingularMatrixError as e:
        train_loss = scores_train["loss"] if "scores_train" in dir() else None
        train_quality = compute_quality(scores_train) if "scores_train" in dir() else None
        val_loss = scores_val["loss"] if "scores_val" in dir() else None
        val_quality = compute_quality(scores_val) if "scores_val" in dir() else None

        success = val_quality is not None and val_quality > 0

        result = {
            "run_id": experiment["run_id"],
            "seed": seed,
            "model": model_name,
            "grid_params": experiment["grid_params"],
            "n_parameters": n_parameters,
            "train_loss": float(train_loss) if train_loss is not None else None,
            "train_quality": float(train_quality) if train_quality is not None else None,
            "val_loss": float(val_loss) if val_loss is not None else None,
            "val_quality": float(val_quality) if val_quality is not None else None,
            "success": success,
            "singular": True,
            "singular_epoch": e.epoch,
        }

        if tracker:
            summary = {**result, "n_parameters": n_parameters}
            tracker.log_summary(summary)

        return result


def is_grid_complete(output_dir: str) -> bool:
    """Check if grid search was already completed."""
    complete_file = os.path.join(output_dir, ".complete")
    return os.path.exists(complete_file)


def mark_grid_complete(output_dir: str) -> None:
    """Mark grid search as complete by creating .complete file."""
    complete_file = os.path.join(output_dir, ".complete")
    with open(complete_file, "w") as f:
        f.write("")


def main():
    parser = argparse.ArgumentParser(description="Run grid search experiments")
    parser.add_argument("--config", type=str, required=True, help="Path to JSON configuration file")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for results (overrides config)",
    )
    parser.add_argument(
        "--tracker",
        type=str,
        default=None,
        choices=["file", "aim", "both", "none"],
        help="Tracking backend (overrides config)",
    )
    parser.add_argument(
        "--tracker-dir",
        type=str,
        default=None,
        help="Directory for file tracker logs (overrides config)",
    )
    parser.add_argument(
        "--aim-repo",
        type=str,
        default=None,
        help="AIM repository URL (e.g., aim://192.168.5.5:5800 for VPN, aim://aimtracking.kosmon.org:443 for non-VPN)",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Show experiment plan without running"
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Print training progress for each run"
    )

    args = parser.parse_args()

    config = load_config(args.config)

    if args.output_dir:
        config["output"]["results_dir"] = args.output_dir
    if args.tracker:
        config["tracker"]["backend"] = args.tracker
    if args.tracker_dir:
        config["tracker"]["tracker_dir"] = args.tracker_dir

    experiments, metadata = create_experiment_plan(config)

    print_experiment_summary(experiments, metadata)

    if args.dry_run:
        print("Dry run complete. No experiments executed.")
        return

    output_dir = config["output"]["results_dir"]
    os.makedirs(output_dir, exist_ok=True)

    if is_grid_complete(output_dir):
        print(f"Grid search already complete. Found .complete file in {output_dir}")
        print("Delete .complete file to re-run experiments.")
        return

    print(f"Loading dataset: {config['dataset']}...")
    data, dataset_info = load_dataset(config["dataset"])

    input_dim = data["X_train"].shape[1]
    output_dim = data["y_train"].shape[1] if data["y_train"].ndim > 1 else 1

    print(f"  Dataset: {dataset_info['name']}")
    print(f"  Train: {dataset_info['n_train']} samples")
    print(f"  Val: {dataset_info['n_val']} samples")
    print(f"  Test: {dataset_info['n_test']} samples")
    print(f"  Features: {dataset_info['n_features']} (+1 bias = {input_dim})")
    print(f"  Task: {dataset_info['task']}")
    print(f"  Device: {DEVICE}")
    print()

    results = []
    skipped_count = 0
    start_time = time.time()

    tracker_backend = config["tracker"]["backend"]
    tracker_dir = config["tracker"].get("tracker_dir", "experiment_logs")
    aim_repo = (
        args.aim_repo
        or os.getenv("AIM_REPO")
        or config["tracker"].get("aim_repo")
        or "aim://aimtracking.kosmon.org:443"
    )

    for i, experiment in enumerate(experiments):
        run_id = experiment["run_id"]

        was_attempted, was_successful, is_singular, error = run_already_completed(
            output_dir, run_id
        )
        if was_attempted:
            if was_successful:
                if args.verbose:
                    print(f"[{i + 1}/{len(experiments)}] {run_id} - SKIPPED (success)")
                skipped_count += 1
                continue
            elif is_singular:
                print(f"[{i + 1}/{len(experiments)}] {run_id} - SKIPPED (singular matrix)")
                skipped_count += 1
                continue
            else:
                err_short = (error[:80] + "...") if error and len(error) > 80 else error
                print(f"[{i + 1}/{len(experiments)}] {run_id} - RETRYING ({err_short})")

        print(f"\n[{i + 1}/{len(experiments)}] Running: {run_id}")

        if tracker_backend != "none":
            tracker = create_tracker(
                experiment_name=experiment["experiment_name"],
                config=experiment["params"],
                backend=tracker_backend,
                output_dir=tracker_dir,
                repo=aim_repo,
                run_name=experiment["run_name"],
            )
        else:
            tracker = None

        result = run_single_experiment(
            experiment, data, input_dim, output_dim, verbose=args.verbose, tracker=tracker
        )

        if tracker:
            tracker.finalize()

        results.append(result)

        if config["output"]["save_individual_runs"]:
            result_file = get_result_filepath(output_dir, experiment["run_id"])
            with open(result_file, "w") as f:
                json.dump(result, f, indent=2)

        if result["success"]:
            quality_name = "R²" if experiment["task"] == "regression" else "Acc"
            train_q = result["train_quality"]
            val_q = result["val_quality"]
            train_str = f"{train_q:.4f}" if train_q is not None else "N/A"
            val_str = f"{val_q:.4f}" if val_q is not None else "N/A"
            singular_marker = " (singular)" if result.get("singular") else ""
            print(
                f"  ✓ Train: {quality_name}={train_str} | Val: {quality_name}={val_str}{singular_marker}"
            )
        else:
            error_msg = result.get("error", "Unknown error")
            print(f"  ✗ FAILED: {error_msg}")

    elapsed_time = time.time() - start_time

    successful_results = [r for r in results if r["success"]]

    print("\n" + "=" * 70)
    print("GRID SEARCH COMPLETE")
    print("=" * 70)
    print(f"Total experiments: {len(experiments)}")
    print(f"Skipped: {skipped_count}")
    print(f"Ran: {len(results)}")
    print(f"Successful: {len(successful_results)}")
    print(f"Failed: {len(results) - len(successful_results)}")
    if len(results) > 0:
        print(f"Time elapsed: {elapsed_time:.1f}s ({elapsed_time / len(results):.1f}s per run)")
    else:
        print(f"Time elapsed: {elapsed_time:.1f}s")
    print()

    if successful_results:
        quality_name = "R²" if config.get("task", "regression") == "regression" else "Accuracy"

        sortable_results = [r for r in successful_results if r["val_quality"] is not None]
        results_sorted = sorted(sortable_results, key=lambda x: x["val_quality"], reverse=True)

        print(f"Top 5 Runs (by val {quality_name}):")
        print()
        for i, result in enumerate(results_sorted[:5]):
            train_q = result["train_quality"]
            val_q = result["val_quality"]
            train_str = f"{train_q:.4f}" if train_q is not None else "N/A"
            val_str = f"{val_q:.4f}" if val_q is not None else "N/A"
            singular_marker = " (singular)" if result.get("singular") else ""
            print(f"{i + 1}. {result['run_id']}{singular_marker}")
            print(f"   Train {quality_name}: {train_str}")
            print(f"   Val {quality_name}: {val_str}")
            print(f"   Params: {result['grid_params']}")
            print()
    elif len(results) == 0:
        print("All experiments already completed. Check results directory for existing results.")
        print()

    summary = {
        "experiment_name": config["experiment_name"],
        "dataset": config["dataset"],
        "task": config.get("task", "regression"),
        "total_experiments": len(results),
        "successful": len(successful_results),
        "failed": len(results) - len(successful_results),
        "elapsed_time": elapsed_time,
        "results": results,
        "metadata": metadata,
        "config": config,
    }

    summary_file = os.path.join(output_dir, "summary.json")
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Results saved to: {output_dir}")
    print(f"Summary: {summary_file}")

    failed_count = len(results) - len(successful_results)
    if failed_count == 0:
        mark_grid_complete(output_dir)
        print("Grid search complete. Marked as .complete")
    else:
        print(f"Grid search has {failed_count} failures. NOT marked as .complete")
        print("Fix failures and re-run to complete.")
    print("=" * 70)


if __name__ == "__main__":
    try:
        main()
    except TrackerError as e:
        print(f"\n[FATAL] Tracker error - terminating job: {e}", file=sys.stderr)
        sys.stderr.flush()
        sys.stdout.flush()
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n[INTERRUPTED] Job cancelled by user", file=sys.stderr)
        sys.exit(130)
    except torch.cuda.OutOfMemoryError as e:
        print(f"\n[FATAL] CUDA out of memory - terminating job: {e}", file=sys.stderr)
        sys.stderr.flush()
        sys.stdout.flush()
        sys.exit(137)
