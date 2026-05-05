import json
import sys
import os
from pathlib import Path

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.dataset_loader import load_dataset

torch.set_default_dtype(torch.float64)


def _compute_r2(y_true, y_pred):
    if y_true.ndim == 1:
        y_true = y_true.unsqueeze(1)
    if y_pred.ndim == 1:
        y_pred = y_pred.unsqueeze(1)
    ss_res = torch.sum((y_true - y_pred) ** 2).item()
    ss_tot = torch.sum((y_true - y_true.mean()) ** 2).item()
    return 0.0 if ss_tot == 0 else 1.0 - ss_res / ss_tot


def _accuracy_onehot(y_true_onehot, y_pred_class):
    return (y_true_onehot.argmax(dim=1) == y_pred_class).float().mean().item()


def run_baseline(dataset_name: str, device: str = "cpu"):
    data, dataset_info = load_dataset(dataset_name, device=device)
    task = dataset_info["task"]

    y_train = data["y_train"]
    y_val = data["y_val"]
    y_test = data["y_test"]

    if task == "regression":
        train_mean = float(y_train.mean())

        val_pred = torch.full((y_val.shape[0], 1), train_mean, dtype=y_val.dtype, device=y_val.device)
        test_pred = torch.full((y_test.shape[0], 1), train_mean, dtype=y_test.dtype, device=y_test.device)

        train_quality = _compute_r2(y_train, torch.full((y_train.shape[0], 1), train_mean, dtype=y_train.dtype, device=y_train.device))
        val_quality = _compute_r2(y_val, val_pred)
        test_quality = _compute_r2(y_test, test_pred)

        train_loss = float(((y_train - train_mean) ** 2).mean().item())
        val_loss = float(((y_val - val_pred) ** 2).mean().item())
        test_loss = float(((y_test - test_pred) ** 2).mean().item())

        summary = {"train_mean": train_mean}

    else:
        majority_class = int(y_train.sum(dim=0).argmax().item())

        n_cls = y_train.shape[1]

        def _logits(n_samples):
            l = torch.zeros(n_samples, n_cls, dtype=y_train.dtype, device=y_train.device)
            l[:, majority_class] = 1.0
            return l

        train_logits = _logits(y_train.shape[0])
        val_logits = _logits(y_val.shape[0])
        test_logits = _logits(y_test.shape[0])

        train_quality = _accuracy_onehot(y_train, torch.full((y_train.shape[0],), majority_class, dtype=torch.long, device=y_train.device))
        val_quality = _accuracy_onehot(y_val, torch.full((y_val.shape[0],), majority_class, dtype=torch.long, device=y_val.device))
        test_quality = _accuracy_onehot(y_test, torch.full((y_test.shape[0],), majority_class, dtype=torch.long, device=y_test.device))

        train_loss = float(torch.nn.functional.cross_entropy(train_logits, y_train).item())
        val_loss = float(torch.nn.functional.cross_entropy(val_logits, y_val).item())
        test_loss = float(torch.nn.functional.cross_entropy(test_logits, y_test).item())

        summary = {"majority_class": majority_class}

    result = {
        "success": True,
        "singular": False,
        "oom_error": False,
        "train_loss": train_loss,
        "train_quality": train_quality,
        "val_loss": val_loss,
        "val_quality": val_quality,
        "test_loss": test_loss,
        "test_quality": test_quality,
        "best_epoch": 0,
        "metrics_log": [{
            "epoch": 0,
            "train_loss": train_loss,
            "train_quality": train_quality,
            "val_loss": val_loss,
            "val_quality": val_quality,
            "test_loss": test_loss,
            "test_quality": test_quality,
        }],
        "gpu_memory": {"cuda_available": torch.cuda.is_available()},
        "dataset_info": dataset_info,
        "baseline_summary": summary,
    }

    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Baseline mean/mode predictor")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name (e.g. abalone, iris)")
    parser.add_argument("--device", type=str, default="cpu", help="Device (cpu or cuda)")
    args = parser.parse_args()

    result = run_baseline(args.dataset, device=args.device)

    out_dir = Path("test_outputs/mean_baseline") / args.dataset
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "results.json"

    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, default=str)

    task = result["dataset_info"]["task"]
    print("Dataset: {} ({})".format(args.dataset, task))
    print("  train_loss:   {:.6f}".format(result["train_loss"]))
    print("  train_quality: {:.6f}".format(result["train_quality"]))
    print("  val_loss:     {:.6f}".format(result["val_loss"]))
    print("  val_quality:   {:.6f}".format(result["val_quality"]))
    print("  test_loss:     {:.6f}".format(result["test_loss"]))
    print("  test_quality:  {:.6f}".format(result["test_quality"]))
    print("Saved to {}".format(out_path))
