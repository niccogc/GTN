#!/usr/bin/env python
"""
Plot y_pred vs y_true on val+test data from a saved model.

Usage:
    python plot_predictions.py outputs/path/to/run/
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import quimb
import torch

from model.base.NTN import NTN
from model.base.NTN_Ensemble import NTN_Ensemble
from model.standard import CPDA, LMPO2, MMPO2, MPO2, TNML_P, TNML_F, BosonMPS
from model.typeI import (
    CPDATypeI, LMPO2TypeI, MMPO2TypeI, MPO2TypeI,
    CPDATypeI_GTN, LMPO2TypeI_GTN, MMPO2TypeI_GTN, MPO2TypeI_GTN,
)
from model.utils import create_inputs
from utils.dataset_loader import load_dataset
from utils.device_utils import DEVICE, move_data_to_device

torch.set_default_dtype(torch.float64)

NTN_MODELS = {
    "MPO2": MPO2, "LMPO2": LMPO2, "MMPO2": MMPO2, "CPDA": CPDA,
    "MPO2TypeI": MPO2TypeI, "LMPO2TypeI": LMPO2TypeI,
    "MMPO2TypeI": MMPO2TypeI, "CPDATypeI": CPDATypeI,
    "TNML_P": TNML_P, "TNML_F": TNML_F,
}
GTN_TYPEI_MODELS = {
    "MPO2TypeI": MPO2TypeI_GTN, "LMPO2TypeI": LMPO2TypeI_GTN,
    "MMPO2TypeI": MMPO2TypeI_GTN, "CPDATypeI": CPDATypeI_GTN,
}
GTN_ONLY_MODELS = {"BosonMPS": BosonMPS}


def make_prediction(run_dir: Path, data: dict) -> tuple:
    """Run model on val+test data. Returns (y_true, y_pred) arrays."""
    with open(run_dir / "results.json") as f:
        results = json.load(f)
    cfg = results["config"]
    model_name = cfg["model"]["name"]
    trainer_type = cfg["trainer"]["type"]
    is_typei = model_name.endswith("TypeI")

    raw_feature_count = data["X_train"].shape[1]
    input_dim = raw_feature_count + 1
    output_dim = 1

    if trainer_type == "gtn":
        model_file = run_dir / "model.pt"
        gtn_cls = GTN_TYPEI_MODELS.get(model_name) or GTN_ONLY_MODELS.get(model_name)
        params = {"phys_dim": input_dim, "output_dim": output_dim, "bond_dim": cfg["model"]["bond_dim"]}
        if is_typei or model_name in GTN_ONLY_MODELS:
            params["max_sites" if is_typei else "L"] = cfg["model"]["L"]
        gtn = gtn_cls(**params)
        gtn.load_state_dict(torch.load(model_file, weights_only=True))
        gtn.to(DEVICE)
        gtn.eval()

        y_true_all, y_pred_all = [], []
        for split in ["val", "test"]:
            X = data[f"X_{split}"]
            y = data[f"y_{split}"]
            X_enc = torch.cat([X, torch.ones(X.shape[0], 1, dtype=X.dtype, device=X.device)], dim=1)
            with torch.no_grad():
                pred = gtn(X_enc).cpu().numpy().flatten()
            y_true_all.extend(y.cpu().numpy().flatten())
            y_pred_all.extend(pred)
        return np.array(y_true_all), np.array(y_pred_all)

    else:
        model_file = run_dir / "model.joblib"
        tn_data = quimb.load_from_disk(model_file)
        model_cls = NTN_MODELS[model_name]

        if is_typei:
            model = model_cls(max_sites=cfg["model"]["L"], bond_dim=cfg["model"]["bond_dim"], phys_dim=input_dim, output_dim=output_dim)
            model.tns = tn_data
            y_true_all, y_pred_all = [], []
            for split in ["val", "test"]:
                X = data[f"X_{split}"]
                y = data[f"y_{split}"]
                dummy_y = torch.zeros(X.shape[0], 1)
                ntn = NTN_Ensemble(tns=model.tns, input_dims_list=model.input_dims_list, input_labels_list=model.input_labels_list, output_dims=model.output_dims, loss=None, X_train=X, y_train=dummy_y, batch_size=X.shape[0])
                pred = ntn.forward()
                y_pred_all.extend(ntn._to_torch(pred).cpu().numpy().flatten())
                y_true_all.extend(y.cpu().numpy().flatten())
            return np.array(y_true_all), np.array(y_pred_all)
        else:
            model = model_cls(L=cfg["model"]["L"], bond_dim=cfg["model"]["bond_dim"], phys_dim=input_dim, output_dim=output_dim)
            model.tn = tn_data
            y_true_all, y_pred_all = [], []
            for split in ["val", "test"]:
                X = data[f"X_{split}"]
                y = data[f"y_{split}"]
                dummy_y = torch.zeros(X.shape[0], 1)
                loader = create_inputs(X=X, y=dummy_y, input_labels=model.input_labels, output_labels=model.output_dims, batch_size=X.shape[0], append_bias=True)
                ntn = NTN(tn=model.tn, output_dims=model.output_dims, input_dims=model.input_dims, loss=None, data_stream=loader)
                pred = ntn.forward(model.tn, loader.data_mu, sum_over_batch=False)
                y_pred_all.extend(ntn._to_torch(pred).cpu().numpy().flatten())
                y_true_all.extend(y.cpu().numpy().flatten())
            return np.array(y_true_all), np.array(y_pred_all)


def plot(run_dir: Path, y_true: np.ndarray, y_pred: np.ndarray):
    with open(run_dir / "results.json") as f:
        results = json.load(f)
    model_name = results["config"]["model"]["name"]

    y_min, y_max = y_true.min(), y_true.max()
    margin = 0.05 * (y_max - y_min)
    lims = [y_min - margin, y_max + margin]

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(y_true, y_pred, alpha=0.5, s=20)
    ax.plot(lims, lims, "k--", lw=1, label="y=x")
    ax.legend()
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel("True Values")
    ax.set_ylabel("Predicted Values")
    ax.set_aspect("equal")

    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    ax.set_title(f"{model_name}  R²={r2:.4f}  RMSE={rmse:.4f}")

    plt.tight_layout()

    plots_dir = Path("plots")
    plots_dir.mkdir(exist_ok=True)
    run_name = str(run_dir.relative_to("outputs")).replace("/", "_") if "outputs" in run_dir.parts else run_dir.name
    out = plots_dir / f"{run_name}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved plot to {out}")
    plt.close()


def resolve_csv_path(cfg: dict, cli_csv_path: str = None) -> str | None:
    if cli_csv_path:
        return cli_csv_path
    stored = cfg["dataset"].get("csv_path")
    if stored and Path(stored).exists():
        return stored
    # Stored path might be a CV temp file - try csvs/ with original name
    if stored:
        stem = Path(stored).stem
        candidate = Path("csvs") / f"{stem}.csv"
        if candidate.exists():
            return str(candidate)
    return stored


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path, help="Run directory with model.joblib/model.pt + results.json")
    parser.add_argument("--csv-path", type=str, help="Override CSV path (needed for CV folds)")
    args = parser.parse_args()

    def plot_run(rd):
        with open(rd / "results.json") as f:
            cfg = json.load(f)["config"]
        csv_path = resolve_csv_path(cfg, args.csv_path)
        data, _ = load_dataset(cfg["dataset"]["name"], csv_path=csv_path, task=cfg["dataset"]["task"])
        data = move_data_to_device(data)
        y_true, y_pred = make_prediction(rd, data)
        plot(rd, y_true, y_pred)

    if not args.path.is_dir() or not (args.path / "results.json").exists():
        run_dirs = [d for d in args.path.iterdir() if d.is_dir() and (d / "results.json").exists()]
        if not run_dirs:
            print(f"No run directories found in {args.path}")
            sys.exit(1)
        for rd in sorted(run_dirs):
            try:
                plot_run(rd)
            except Exception as e:
                print(f"Error processing {rd}: {e}")
        return

    plot_run(args.path)


if __name__ == "__main__":
    main()
