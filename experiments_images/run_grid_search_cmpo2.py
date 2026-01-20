# type: ignore
"""
Grid search experiment runner for CMPO2_GTN on image datasets.
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
from experiments_images.image_dataset_loader import load_image_dataset
from experiments.trackers import create_tracker, TrackerError

from testing.CMPO2_models import CMPO2, CMPO2_GTN
from testing.CMPO3_models import CMPO3, CMPO3_GTN

torch.set_default_dtype(torch.float64)

from experiments.device_utils import DEVICE


def get_result_filepath(output_dir: str, run_id: str) -> str:
    return os.path.join(output_dir, f"{run_id}.json")


def run_already_completed(output_dir: str, run_id: str) -> tuple[bool, bool, bool, str | None]:
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


def create_model(params: dict, dataset_info: dict, model_type: str = "cmpo2"):
    n_sites = params["n_sites"]
    rank_pixel = params["rank_pixel"]
    rank_patch = params["rank_patch"]
    init_strength = params.get("init_strength", 0.01)

    n_patches = dataset_info["n_patches"]
    pixels_per_patch = dataset_info["pixels_per_patch"]
    output_dim = dataset_info["n_classes"]

    if model_type == "cmpo3":
        rank_channel = params.get("rank_channel", 4)
        n_channels = dataset_info["n_channels"]

        cmpo3 = CMPO3(
            n_sites=n_sites,
            channel_dim=n_channels,
            pixel_dim=pixels_per_patch,
            patch_dim=n_patches,
            rank_channel=rank_channel,
            rank_pixel=rank_pixel,
            rank_patch=rank_patch,
            output_dim=output_dim,
            init_strength=init_strength,
        )
        return CMPO3_GTN(cmpo3)
    else:
        cmpo2 = CMPO2(
            n_sites=n_sites,
            pixel_dim=pixels_per_patch,
            patch_dim=n_patches,
            rank_pixel=rank_pixel,
            rank_patch=rank_patch,
            output_dim=output_dim,
            init_strength=init_strength,
        )
        return CMPO2_GTN(cmpo2)


def run_single_experiment(
    experiment: dict,
    data: dict,
    dataset_info: dict,
    verbose: bool = False,
    tracker=None,
):
    params = experiment["params"]
    seed = experiment["seed"]

    torch.manual_seed(seed)
    np.random.seed(seed)

    criterion = nn.CrossEntropyLoss()

    n_epochs = params.get("n_epochs", 50)
    batch_size = params.get("batch_size", 32)
    lr = params.get("lr", 0.001)
    weight_decay = params.get("weight_decay", 0.01)
    optimizer_name = params.get("optimizer", "adam").lower()
    patience = params.get("patience", None)
    min_delta = params.get("min_delta", 0.0)

    model_type = dataset_info.get("model_type", "cmpo2")
    model = create_model(params, dataset_info, model_type)
    model = model.to(DEVICE)

    if optimizer_name == "adam":
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == "adamw":
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay, momentum=0.9)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")

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
        model.eval()
        total_loss = 0
        correct, total = 0, 0
        with torch.no_grad():
            for batch_data, batch_target in loader:
                batch_data, batch_target = batch_data.to(DEVICE), batch_target.to(DEVICE)
                output = model(batch_data)
                loss = criterion(output, batch_target)
                total_loss += loss.item() * batch_data.size(0)

                pred = output.argmax(dim=1)
                target_labels = batch_target.argmax(dim=1)
                correct += (pred == target_labels).sum().item()
                total += batch_target.size(0)

        avg_loss = total_loss / len(loader.dataset)
        accuracy = correct / total if total > 0 else 0.0
        return avg_loss, accuracy

    model_name = "CMPO3_GTN" if model_type == "cmpo3" else "CMPO2_GTN"
    n_parameters = sum(p.numel() for p in model.parameters())

    if tracker:
        hparams = {
            "seed": seed,
            "model": model_name,
            "dataset": experiment["dataset"],
            "n_parameters": n_parameters,
            **params,
        }
        tracker.log_hparams(hparams)

    best_val_accuracy = 0.0
    best_train_loss = float("inf")
    best_epoch = -1
    patience_counter = 0
    stopped_early = False

    for epoch in range(n_epochs):
        model.train()
        train_loss = 0.0

        for batch_data, batch_target in train_loader:
            batch_data, batch_target = batch_data.to(DEVICE), batch_target.to(DEVICE)
            optimizer.zero_grad()
            output = model(batch_data)
            loss = criterion(output, batch_target)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * batch_data.size(0)

        train_loss /= len(train_loader.dataset)
        _, train_accuracy = evaluate(train_loader)
        val_loss, val_accuracy = evaluate(val_loader)

        if train_loss < best_train_loss - min_delta:
            best_train_loss = train_loss
            patience_counter = 0
        else:
            patience_counter += 1

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_epoch = epoch

        if patience is not None and patience_counter >= patience:
            if verbose:
                print(f"\n  Early stopping at epoch {epoch + 1}")
            stopped_early = True
            break

        if tracker:
            metrics = {
                "train_loss": train_loss,
                "train_quality": train_accuracy,
                "val_loss": val_loss,
                "val_quality": val_accuracy,
                "patience_counter": patience_counter,
            }
            tracker.log_metrics(metrics, step=epoch)

        if verbose and (epoch % 10 == 0 or epoch == n_epochs - 1):
            print(
                f"  Epoch {epoch + 1:3d} | Train Loss: {train_loss:.4f} | "
                f"Train Acc: {train_accuracy:.4f} | Val Acc: {val_accuracy:.4f}"
            )

    train_loss, train_accuracy = evaluate(train_loader)
    test_loss, test_accuracy = evaluate(test_loader)

    result = {
        "run_id": experiment["run_id"],
        "seed": seed,
        "model": model_name,
        "dataset": experiment["dataset"],
        "task": "classification",
        "params": params,
        "n_parameters": n_parameters,
        "train_loss": float(train_loss),
        "train_accuracy": float(train_accuracy),
        "val_loss": float(val_loss),
        "val_accuracy": float(best_val_accuracy),
        "test_loss": float(test_loss),
        "test_accuracy": float(test_accuracy),
        "best_epoch": best_epoch,
        "stopped_early": stopped_early,
        "patience_counter": patience_counter,
        "success": True,
    }

    if tracker:
        tracker.log_summary(
            {
                "test_quality": test_accuracy,
                "test_loss": test_loss,
                "best_val_quality": best_val_accuracy,
                "n_parameters": n_parameters,
            }
        )

    return result


def generate_run_id(grid_params: dict, seed: int, model_type: str = "cmpo2") -> str:
    model_name = "CMPO3_GTN" if model_type == "cmpo3" else "CMPO2_GTN"
    parts = [model_name]

    if "n_patches" in grid_params:
        parts.append(f"np{grid_params['n_patches']}")
    if "n_sites" in grid_params:
        parts.append(f"ns{grid_params['n_sites']}")
    if "rank_channel" in grid_params and model_type == "cmpo3":
        parts.append(f"rc{grid_params['rank_channel']}")
    if "rank_pixel" in grid_params:
        parts.append(f"rp{grid_params['rank_pixel']}")
    if "rank_patch" in grid_params:
        parts.append(f"rpa{grid_params['rank_patch']}")
    if "init_strength" in grid_params:
        val = grid_params["init_strength"]
        if val < 0.01:
            parts.append(f"init{val:.0e}".replace("-", "m"))
        else:
            parts.append(f"init{val:.3f}".rstrip("0").rstrip("."))
    if "lr" in grid_params:
        val = grid_params["lr"]
        parts.append(f"lr{val:.0e}".replace("-", "m"))
    if "weight_decay" in grid_params:
        val = grid_params["weight_decay"]
        parts.append(f"wd{val:.0e}".replace("-", "m"))

    parts.append(f"seed{seed}")

    return "-".join(parts)


def generate_run_name(grid_params: dict, seed: int, model_type: str = "cmpo2") -> str:
    return generate_run_id(grid_params, seed, model_type)


def create_experiment_plan_cmpo2(config: dict):
    from experiments.config_parser import expand_parameter_grid

    grid_combinations = expand_parameter_grid(config["parameter_grid"])
    model_type = config.get("model_type", "cmpo2")

    seeds = config["fixed_params"].get("seeds", [0])
    if not isinstance(seeds, list):
        seeds = [seeds]

    experiments = []

    for grid_params in grid_combinations:
        full_params = {}
        full_params.update(config.get("fixed_params", {}))
        full_params.update(grid_params)

        for seed in seeds:
            experiment = {
                "experiment_name": config["experiment_name"],
                "dataset": config["dataset"],
                "task": "classification",
                "params": full_params,
                "seed": seed,
                "run_name": generate_run_name(grid_params, seed, model_type),
                "run_id": generate_run_id(grid_params, seed, model_type),
                "grid_params": grid_params,
                "tracker": config.get("tracker", {}),
                "output": config.get("output", {}),
            }
            experiments.append(experiment)

    metadata = {
        "total_experiments": len(experiments),
        "grid_size": len(grid_combinations),
        "n_seeds": len(seeds),
        "seeds": seeds,
        "parameter_grid": config["parameter_grid"],
        "fixed_params": config.get("fixed_params", {}),
    }

    return experiments, metadata


def main():
    parser = argparse.ArgumentParser(description="Run CMPO2_GTN grid search on image datasets")
    parser.add_argument("--config", type=str, required=True, help="Path to JSON config file")
    parser.add_argument(
        "--output-dir", type=str, required=True, help="Output directory for results"
    )
    parser.add_argument(
        "--tracker", type=str, default="file", choices=["file", "aim", "both", "none"]
    )
    parser.add_argument("--tracker-dir", type=str, default="experiment_logs")
    parser.add_argument("--aim-repo", type=str, default=None)
    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args()

    config = load_config(args.config)
    experiment_plan, metadata = create_experiment_plan_cmpo2(config)

    if args.tracker == "file" and "tracker" in config:
        args.tracker = config["tracker"].get("backend", "file")
    if args.tracker_dir == "experiment_logs" and "tracker" in config:
        args.tracker_dir = config["tracker"].get("tracker_dir", "experiment_logs")
    if args.aim_repo is None and "tracker" in config:
        args.aim_repo = config["tracker"].get("aim_repo", None)

    os.makedirs(args.output_dir, exist_ok=True)

    dataset_name = config["dataset"]
    model_type = config.get("model_type", "cmpo2")
    n_train = config.get("n_train", None)
    n_val = config.get("n_val", None)
    n_test = config.get("n_test", None)

    n_patches_values = config["parameter_grid"].get("n_patches", [4])
    if not isinstance(n_patches_values, list):
        n_patches_values = [n_patches_values]

    data_cache = {}
    for n_patches in n_patches_values:
        print(f"\nLoading dataset: {dataset_name} (n_patches={n_patches}, model={model_type})")
        data, dataset_info = load_image_dataset(
            dataset_name,
            n_patches=n_patches,
            n_train=n_train,
            n_val=n_val,
            n_test=n_test,
            model_type=model_type,
        )
        data_cache[n_patches] = (data, dataset_info)
        shape_str = f"({n_patches}, {dataset_info['pixels_per_patch']})"
        if model_type == "cmpo3":
            shape_str = (
                f"({n_patches}, {dataset_info['pixels_per_patch']}, {dataset_info['n_channels']})"
            )
        print(
            f"  Train: {dataset_info['n_train']}, Val: {dataset_info['n_val']}, Test: {dataset_info['n_test']}"
        )
        print(f"  Shape: {shape_str}, Classes: {dataset_info['n_classes']}")

    print(f"  Device: {DEVICE}")

    print_experiment_summary(experiment_plan, metadata)

    results = []
    completed_count = 0
    skipped_count = 0
    failed_count = 0

    start_time = time.time()

    for idx, experiment in enumerate(experiment_plan, 1):
        run_id = experiment["run_id"]

        was_attempted, was_successful, is_singular, error = run_already_completed(
            args.output_dir, run_id
        )
        if was_attempted:
            if was_successful:
                if args.verbose:
                    print(f"[{idx}/{len(experiment_plan)}] {run_id} - SKIPPED (success)")
                skipped_count += 1
                continue
            elif is_singular:
                print(f"[{idx}/{len(experiment_plan)}] {run_id} - SKIPPED (singular)")
                skipped_count += 1
                continue
            else:
                err_short = (error[:80] + "...") if error and len(error) > 80 else error
                print(f"[{idx}/{len(experiment_plan)}] {run_id} - RETRYING ({err_short})")

        n_patches = experiment["params"].get("n_patches", 4)
        data, dataset_info = data_cache[n_patches]

        print(f"\n[{idx}/{len(experiment_plan)}] Running: {run_id}")

        aim_repo = (
            args.aim_repo
            or config.get("tracker", {}).get("aim_repo")
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
                dataset_info=dataset_info,
                verbose=args.verbose,
                tracker=tracker,
            )

            result_file = get_result_filepath(args.output_dir, run_id)
            with open(result_file, "w") as f:
                json.dump(result, f, indent=2)

            results.append(result)
            completed_count += 1

            print(f"  Test Acc={result['test_accuracy']:.4f}")

        except TrackerError:
            raise

        except Exception as e:
            import traceback

            error_tb = traceback.format_exc()
            error_str = str(e)
            error_short = error_str[:200]

            print(f"  FAILED: {error_short}")
            failed_count += 1

            error_result = {
                "run_id": run_id,
                "success": False,
                "singular": False,
                "error": error_str,
                "traceback": error_tb,
            }

            result_file = get_result_filepath(args.output_dir, run_id)
            with open(result_file, "w") as f:
                json.dump(error_result, f, indent=2)

        finally:
            if tracker and hasattr(tracker, "close"):
                tracker.close()

    elapsed_time = time.time() - start_time

    print("\n" + "=" * 70)
    print("GRID SEARCH SUMMARY")
    print("=" * 70)
    print(f"Total runs: {len(experiment_plan)}")
    print(f"Completed: {completed_count}")
    print(f"Skipped: {skipped_count}")
    print(f"Failed: {failed_count}")
    print(f"Time: {elapsed_time:.1f}s")

    if results:
        results_sorted = sorted(results, key=lambda r: r["test_accuracy"], reverse=True)

        print(f"\nTop 5 by Test Accuracy:")
        for i, r in enumerate(results_sorted[:5], 1):
            print(f"\n{i}. {r['run_id']}")
            print(f"   Test Acc: {r['test_accuracy'] * 100:.2f}%")
            print(f"   Seed: {r['seed']}")

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
