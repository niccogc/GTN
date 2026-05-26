# type: ignore
"""
Dataset loader utility for production training scripts.
Supports UCI ML Repository (ucirepo) and local CSV files (csvs/).
"""

import torch
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.load_ucirepo import get_ucidata, datasets as uci_datasets
from model.load_from_csv import get_csvdata


def load_dataset(
    dataset_name,
    csv_path=None,
    task=None,
    n_samples=None,
    seed=0,
    val_split=0.2,
    test_split=0.2,
    device="cpu",
    cap=50,
):
    """
    Load a dataset by name or from a CSV file.

    Two modes:
      1. UCI dataset: dataset_name is a UCI dataset name (e.g., 'abalone', 'iris').
         csv_path and task are ignored (task comes from the UCI registry).
      2. CSV file: if csv_path is provided, loads from that CSV.
         dataset_name is used for metadata only; task is required.

    Note: Train/val/test splits are FIXED (seed=42) for reproducibility and fair
    comparison across experiments. The experiment seed controls model initialization,
    not data splits. This is standard practice - it isolates model variance from
    data variance.

    Args:
        dataset_name: Name of dataset (UCI name or label for CSV)
        csv_path: Path to CSV file (absolute, or relative to csvs/)
        task: Task type for CSV data ("regression" or "classification", default: "regression")
        n_samples: Number of samples to use (None = use all) - NOT IMPLEMENTED
        seed: Unused (splits are fixed; experiment seed controls model init)
        val_split: Unused (fixed 70/15/15 split)
        test_split: Unused (fixed 70/15/15 split)
        device: Device to load tensors on (default: 'cpu')
        cap: Maximum number of features after one-hot encoding (default: 50)

    Returns:
        data: dict with keys 'X_train', 'y_train', 'X_val', 'y_val', 'X_test', 'y_test'
        dataset_info: dict with metadata
    """
    # ── CSV path provided → load from file ──
    if csv_path is not None:
        _task = task if task is not None else "regression"
        X_train, y_train, X_val, y_val, X_test, y_test = get_csvdata(
            csv_path, task=_task, device=device, cap=cap
        )
        source = os.path.basename(csv_path)
        dataset_id = None

    # ── UCI dataset ──
    else:
        uci_dataset_map = {name: (dataset_id, task) for name, dataset_id, task in uci_datasets}

        if dataset_name not in uci_dataset_map:
            raise ValueError(
                f"Dataset '{dataset_name}' not found. "
                f"Available datasets: {list(uci_dataset_map.keys())}"
            )

        dataset_id, _task = uci_dataset_map[dataset_name]

        X_train, y_train, X_val, y_val, X_test, y_test = get_ucidata(
            dataset_id=dataset_id, task=_task, device=device, cap=cap
        )
        source = "ucirepo"

    if _task == "regression":
        if y_train.ndim == 1:
            y_train = y_train.unsqueeze(1)
        if y_val.ndim == 1:
            y_val = y_val.unsqueeze(1)
        if y_test.ndim == 1:
            y_test = y_test.unsqueeze(1)
    elif _task == "classification":
        n_classes = len(torch.unique(y_train))
        y_train_onehot = torch.zeros(len(y_train), n_classes, dtype=torch.float64, device=device)
        y_train_onehot[torch.arange(len(y_train)), y_train] = 1.0
        y_train = y_train_onehot

        y_val_onehot = torch.zeros(len(y_val), n_classes, dtype=torch.float64, device=device)
        y_val_onehot[torch.arange(len(y_val)), y_val] = 1.0
        y_val = y_val_onehot

        y_test_onehot = torch.zeros(len(y_test), n_classes, dtype=torch.float64, device=device)
        y_test_onehot[torch.arange(len(y_test)), y_test] = 1.0
        y_test = y_test_onehot

    data = {
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
        "X_test": X_test,
        "y_test": y_test,
    }

    dataset_info = {
        "name": dataset_name,
        "source": source,
        "dataset_id": dataset_id,
        "n_samples": len(X_train) + len(X_val) + len(X_test),
        "n_train": len(X_train),
        "n_val": len(X_val),
        "n_test": len(X_test),
        "n_features": X_train.shape[1],
        "task": _task,
    }

    if _task == "classification":
        dataset_info["n_classes"] = n_classes

    return data, dataset_info
