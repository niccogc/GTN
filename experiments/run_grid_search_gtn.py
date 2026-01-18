# type: ignore
"""
Grid search experiment runner for GTN (gradient-based training).
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
import torch.nn as nn
import torch.optim as optim
import numpy as np

from experiments.config_parser import load_config, create_experiment_plan, print_experiment_summary
from experiments.dataset_loader import load_dataset
from experiments.trackers import create_tracker

from model.GTN import GTN
from model.MPO2_models import MPO2, LMPO2, MMPO2
from model.typeI import MPO2TypeI_GTN, LMPO2TypeI_GTN, MMPO2TypeI_GTN

torch.set_default_dtype(torch.float64)

from experiments.device_utils import DEVICE, move_tn_to_device


class MPO2GTN(GTN):
    """GTN wrapper for MPO2 models."""

    def construct_nodes(self, x):
        import quimb.tensor as qt

        input_nodes = []
        for label in self.input_dims:
            a = qt.Tensor(x, inds=["s", label], tags=f"Input_{label}")
            input_nodes.append(a)
        return input_nodes


def get_result_filepath(output_dir: str, run_id: str) -> str:
    """Get filepath for individual run result."""
    return os.path.join(output_dir, f"{run_id}.json")


def run_already_completed(output_dir: str, run_id: str) -> tuple[bool, bool, str | None]:
    """
    Check if a run has already been attempted.

    Returns:
        (was_attempted, was_successful, error_message)
    """
    result_file = get_result_filepath(output_dir, run_id)

    if not os.path.exists(result_file):
        return False, False, None

    try:
        with open(result_file, "r") as f:
            result = json.load(f)
        success = result.get("success", False)
        error = result.get("error", None)
        return True, success, error
    except:
        return False, False, None


def create_model(model_name: str, params: dict, input_dim: int, output_dim: int):
    """Create model instance based on model name and parameters."""

    if model_name == "MPO2":
        return MPO2(
            L=params["L"],
            bond_dim=params["bond_dim"],
            phys_dim=input_dim,
            output_dim=output_dim,
            output_site=params.get("output_site", 1),
            init_strength=params.get("init_strength", 0.001),
        )

    elif model_name == "LMPO2":
        return LMPO2(
            L=params["L"],
            bond_dim=params["bond_dim"],
            phys_dim=input_dim,
            output_dim=output_dim,
            rank=params.get("rank", 5),
            output_site=params.get("output_site", 1),
            init_strength=params.get("init_strength", 0.001),
        )

    elif model_name == "MMPO2":
        return MMPO2(
            L=params["L"],
            bond_dim=params["bond_dim"],
            phys_dim=input_dim,
            output_dim=output_dim,
            rank=params.get("rank", 5),
            output_site=params.get("output_site", 1),
            init_strength=params.get("init_strength", 0.001),
        )

    elif model_name == "MPO2TypeI_GTN":
        return MPO2TypeI_GTN(
            max_sites=params["L"],
            bond_dim=params["bond_dim"],
            phys_dim=input_dim,
            output_dim=output_dim,
            output_site=params.get("output_site", 1),
            init_strength=params.get("init_strength", 0.001),
        )

    elif model_name == "LMPO2TypeI_GTN":
        return LMPO2TypeI_GTN(
            max_sites=params["L"],
            bond_dim=params["bond_dim"],
            phys_dim=input_dim,
            reduced_dim=params.get("rank", 5),
            output_dim=output_dim,
            output_site=params.get("output_site", 1),
            init_strength=params.get("init_strength", 0.001),
        )

    elif model_name == "MMPO2TypeI_GTN":
        return MMPO2TypeI_GTN(
            max_sites=params["L"],
            bond_dim=params["bond_dim"],
            phys_dim=input_dim,
            output_dim=output_dim,
            output_site=params.get("output_site", 1),
            init_strength=params.get("init_strength", 0.001),
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
    """Run a single GTN experiment with given parameters."""

    params = experiment["params"]
    seed = experiment["seed"]
    task = experiment["task"]

    torch.manual_seed(seed)
    np.random.seed(seed)

    # Setup loss function
    loss_fn_name = params.get("loss_fn", None)
    if loss_fn_name:
        if loss_fn_name == "mse":
            criterion = nn.MSELoss()
        elif loss_fn_name == "mae":
            criterion = nn.L1Loss()
        elif loss_fn_name == "huber":
            criterion = nn.HuberLoss()
        elif loss_fn_name == "cross_entropy":
            criterion = nn.CrossEntropyLoss()
        else:
            raise ValueError(f"Unknown loss function: {loss_fn_name}")
    else:
        if task == "regression":
            criterion = nn.MSELoss()
        else:
            criterion = nn.CrossEntropyLoss()

    # Get training parameters
    n_epochs = params.get("n_epochs", 50)
    batch_size = params.get("batch_size", 32)
    lr = params.get("lr", 0.001)
    weight_decay = params.get("weight_decay", 0.01)
    optimizer_name = params.get("optimizer", "adam").lower()
    patience = params.get("patience", None)
    min_delta = params.get("min_delta", 0.0)

    model_name = params["model"]
    base_model = create_model(model_name, params, input_dim, output_dim)

    if model_name.endswith("_GTN"):
        gtn_model = base_model
    else:
        gtn_model = MPO2GTN(tn=base_model.tn, output_dims=["out"], input_dims=base_model.input_dims)

    gtn_model = gtn_model.to(DEVICE)

    # Create optimizer
    if optimizer_name == "adam":
        optimizer = optim.Adam(gtn_model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == "adamw":
        optimizer = optim.AdamW(gtn_model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == "sgd":
        optimizer = optim.SGD(
            gtn_model.parameters(), lr=lr, weight_decay=weight_decay, momentum=0.9
        )
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")

    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(data["X_train"], data["y_train"]),
        batch_size=batch_size,
        shuffle=True,
    )
    val_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(data["X_val"], data["y_val"]),
        batch_size=batch_size,
        shuffle=False,
    )
    test_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(data["X_test"], data["y_test"]),
        batch_size=batch_size,
        shuffle=False,
    )

    def evaluate(loader):
        gtn_model.eval()
        total_loss = 0

        if task == "regression":
            all_preds, all_targets = [], []
            with torch.no_grad():
                for batch_data, batch_target in loader:
                    batch_data, batch_target = batch_data.to(DEVICE), batch_target.to(DEVICE)
                    output = gtn_model(batch_data)
                    loss = criterion(output, batch_target)
                    total_loss += loss.item() * batch_data.size(0)
                    all_preds.append(output.cpu())
                    all_targets.append(batch_target.cpu())

            preds = torch.cat(all_preds, dim=0)
            targets = torch.cat(all_targets, dim=0)
            ss_res = torch.sum((targets - preds) ** 2).item()
            ss_tot = torch.sum((targets - targets.mean()) ** 2).item()
            quality = 1 - ss_res / ss_tot if ss_tot > 0 else float("-inf")
        else:
            correct, total = 0, 0
            with torch.no_grad():
                for batch_data, batch_target in loader:
                    batch_data, batch_target = batch_data.to(DEVICE), batch_target.to(DEVICE)
                    output = gtn_model(batch_data)
                    loss = criterion(output, batch_target)
                    total_loss += loss.item() * batch_data.size(0)

                    pred = output.argmax(dim=1)
                    target_labels = batch_target.argmax(dim=1)
                    correct += (pred == target_labels).sum().item()
                    total += batch_target.size(0)

            quality = correct / total if total > 0 else 0.0

        avg_loss = total_loss / len(loader.dataset)
        return avg_loss, quality

    # Log hyperparameters
    if tracker:
        hparams = {"seed": seed, "model": model_name, "dataset": experiment["dataset"], **params}
        tracker.log_hparams(hparams)

    # Training loop
    best_val_quality = float("-inf")
    best_train_loss = float("inf")
    best_epoch = -1
    patience_counter = 0
    stopped_early = False

    for epoch in range(n_epochs):
        gtn_model.train()
        train_loss = 0.0

        for batch_data, batch_target in train_loader:
            batch_data, batch_target = batch_data.to(DEVICE), batch_target.to(DEVICE)
            optimizer.zero_grad()
            output = gtn_model(batch_data)
            loss = criterion(output, batch_target)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * batch_data.size(0)

        train_loss /= len(train_loader.dataset)
        val_loss, val_quality = evaluate(val_loader)

        # Early stopping based on train loss
        if train_loss < best_train_loss - min_delta:
            best_train_loss = train_loss
            patience_counter = 0
        else:
            patience_counter += 1

        if val_quality > best_val_quality:
            best_val_quality = val_quality
            best_epoch = epoch

        # Early stopping check
        if patience is not None and patience_counter >= patience:
            if verbose:
                print(f"\n⏸ Early stopping triggered at epoch {epoch + 1}")
                print(
                    f"  No improvement in train loss for {patience} epochs (min_delta={min_delta})"
                )
                print(f"  Best train loss: {best_train_loss:.6f}")
            stopped_early = True
            break

        # Log metrics
        if tracker:
            metrics = {
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_quality": val_quality,
                "patience_counter": patience_counter,
            }
            tracker.log_metrics(metrics, step=epoch)

        if verbose and (epoch % 10 == 0 or epoch == n_epochs - 1):
            print(
                f"  Epoch {epoch + 1:3d} | Train Loss: {train_loss:.4f} | Val Quality: {val_quality:.4f} | Patience: {patience_counter}"
            )

    # Final evaluation on all sets
    train_loss, train_quality = evaluate(train_loader)
    test_loss, test_quality = evaluate(test_loader)

    result = {
        "run_id": experiment["run_id"],
        "seed": seed,
        "model": model_name,
        "dataset": experiment["dataset"],
        "task": task,
        "params": params,
        "train_loss": float(train_loss),
        "train_quality": float(train_quality),
        "val_loss": float(val_loss),
        "val_quality": float(best_val_quality),
        "test_loss": float(test_loss),
        "test_quality": float(test_quality),
        "best_epoch": best_epoch,
        "stopped_early": stopped_early,
        "patience_counter": patience_counter,
        "success": True,
    }

    if tracker:
        tracker.log_summary(
            {
                "test_quality": test_quality,
                "test_loss": test_loss,
                "best_val_quality": best_val_quality,
            }
        )

    return result


def main():
    parser = argparse.ArgumentParser(description="Run GTN grid search experiments")
    parser.add_argument("--config", type=str, required=True, help="Path to JSON config file")
    parser.add_argument(
        "--output-dir", type=str, required=True, help="Output directory for results"
    )
    parser.add_argument(
        "--tracker",
        type=str,
        default="file",
        choices=["file", "aim", "both", "none"],
        help="Tracking backend",
    )
    parser.add_argument(
        "--tracker-dir", type=str, default="experiment_logs", help="Directory for file tracker logs"
    )
    parser.add_argument(
        "--aim-repo", type=str, default=None, help="AIM repository (local path or aim://host:port)"
    )
    parser.add_argument("--verbose", action="store_true", help="Print detailed progress")

    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)
    experiment_plan, metadata = create_experiment_plan(config)

    # Override tracker settings from config if not specified via command line
    if args.tracker == "file" and "tracker" in config:
        args.tracker = config["tracker"].get("backend", "file")
    if args.tracker_dir == "experiment_logs" and "tracker" in config:
        args.tracker_dir = config["tracker"].get("tracker_dir", "experiment_logs")
    if args.aim_repo is None and "tracker" in config:
        args.aim_repo = config["tracker"].get("aim_repo", None)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load dataset (shared across all experiments)
    dataset_name = config["dataset"]
    print(f"\nLoading dataset: {dataset_name}")
    data, dataset_info = load_dataset(dataset_name)

    input_dim = data["X_train"].shape[1]

    if config["task"] == "regression":
        output_dim = 1
    else:
        output_dim = dataset_info["n_classes"]

    print(f"  Task: {config['task']}")
    print(f"  Train: {dataset_info['n_train']} samples")
    print(f"  Val: {dataset_info['n_val']} samples")
    print(f"  Test: {dataset_info['n_test']} samples")
    print(f"  Input dim: {input_dim}")
    print(f"  Output dim: {output_dim}")
    print(f"  Device: {DEVICE}")

    # Print experiment summary
    print_experiment_summary(experiment_plan, metadata)

    # Run experiments
    results = []
    completed_count = 0
    skipped_count = 0
    failed_count = 0

    start_time = time.time()

    for idx, experiment in enumerate(experiment_plan, 1):
        run_id = experiment["run_id"]

        was_attempted, was_successful, error = run_already_completed(args.output_dir, run_id)
        if was_attempted:
            if was_successful:
                if args.verbose:
                    print(f"[{idx}/{len(experiment_plan)}] {run_id} - SKIPPED (success)")
            else:
                err_short = (error[:80] + "...") if error and len(error) > 80 else error
                print(f"[{idx}/{len(experiment_plan)}] {run_id} - SKIPPED (failed: {err_short})")
            skipped_count += 1
            continue

        print(f"\n[{idx}/{len(experiment_plan)}] Running: {run_id}")

        # Set up AIM repo with config file having priority over environment
        aim_repo = (
            args.aim_repo
            or config["tracker"].get("aim_repo")
            or os.getenv("AIM_REPO")
            or "aim://aimtracking.kosmon.org:443"
        )

        tracker = create_tracker(
            experiment_name=config["experiment_name"],
            config=experiment,
            backend=args.tracker,
            output_dir=args.tracker_dir,
            repo=aim_repo,
            run_name=experiment["run_name"],
        )

        try:
            result = run_single_experiment(
                experiment=experiment,
                data=data,
                input_dim=input_dim,
                output_dim=output_dim,
                verbose=args.verbose,
                tracker=tracker,
            )

            # Save individual result
            result_file = get_result_filepath(args.output_dir, run_id)
            with open(result_file, "w") as f:
                json.dump(result, f, indent=2)

            results.append(result)
            completed_count += 1

            quality_name = "R²" if config["task"] == "regression" else "Acc"
            print(f"  ✓ SUCCESS: Test {quality_name}={result['test_quality']:.4f}")

        except Exception as e:
            import traceback

            error_tb = traceback.format_exc()
            error_short = str(e)[:200]
            print(f"  ✗ FAILED: {error_short}")
            failed_count += 1

            error_result = {
                "run_id": run_id,
                "success": False,
                "error": str(e),
                "traceback": error_tb,
            }

            result_file = get_result_filepath(args.output_dir, run_id)
            with open(result_file, "w") as f:
                json.dump(error_result, f, indent=2)

        finally:
            if tracker and hasattr(tracker, "close"):
                tracker.close()

    elapsed_time = time.time() - start_time

    # Generate summary
    print("\n" + "=" * 70)
    print("GRID SEARCH SUMMARY")
    print("=" * 70)
    print(f"Total runs: {len(experiment_plan)}")
    print(f"Completed: {completed_count}")
    print(f"Skipped: {skipped_count}")
    print(f"Failed: {failed_count}")
    print(f"Time: {elapsed_time:.1f}s ({elapsed_time / len(experiment_plan):.1f}s per run)")

    if results:
        # Sort by test quality
        results_sorted = sorted(results, key=lambda r: r["test_quality"], reverse=True)

        quality_name = "R²" if config["task"] == "regression" else "Accuracy"
        multiplier = 1 if config["task"] == "regression" else 100

        print(f"\nTop 5 Configurations by Test {quality_name}:")
        for i, r in enumerate(results_sorted[:5], 1):
            print(f"\n{i}. {r['run_id']}")
            print(f"   Test {quality_name}: {r['test_quality'] * multiplier:.2f}")
            print(f"   Model: {r['model']}, Seed: {r['seed']}")
            print(
                f"   Key params: bond_dim={r['params']['bond_dim']}, "
                f"lr={r['params']['lr']}, wd={r['params'].get('weight_decay', 0)}"
            )

        # Save summary
        summary = {
            "config": config,
            "dataset_info": dataset_info,
            "total_runs": len(experiment_plan),
            "completed": completed_count,
            "skipped": skipped_count,
            "failed": failed_count,
            "elapsed_time": elapsed_time,
            "top_configurations": results_sorted[:10],
        }

        summary_file = os.path.join(args.output_dir, "summary.json")
        with open(summary_file, "w") as f:
            json.dump(summary, f, indent=2)

        print(f"\nSummary saved to: {summary_file}")

    print("=" * 70)


if __name__ == "__main__":
    main()
