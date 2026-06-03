# type: ignore
"""
Unified experiment runner with Hydra configuration.

Usage:
    python run.py                                    # defaults (MPO2, iris, NTN)
    python run.py model=lmpo2 dataset=abalone        # override model/dataset
    python run.py trainer=gtn trainer.lr=0.01        # GTN with custom LR
    python run.py --multirun model.bond_dim=4,6,8    # sweep bond dimensions
    python run.py --multirun seed=0,1,2,3,4          # multi-seed
    python run.py model=cmpo2 dataset=mnist          # image classification
"""

import json
import logging
import time
from collections import defaultdict
from pathlib import Path

import hydra
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from omegaconf import DictConfig, OmegaConf

from utils.dataset_loader import load_dataset
from utils.image_dataset_loader import load_image_dataset, load_image_dataset_cnn
from utils.device_utils import DEVICE, move_data_to_device, move_tn_to_device
from utils.tracking import (
    generate_run_id,
    load_tracking_file,
    should_skip_run,
    append_run_result,
    DEFAULT_TRACKING_FILE,
)
from model.base.GTN import GTN
from model.base.NTN import NTN
from model.base.NTN_Ensemble import NTN_Ensemble
from model.base.DMRG import DMRG
from model.exceptions import SingularMatrixError
from model.losses import CrossEntropyLoss, MSELoss
from model.standard import CPDA, LMPO2, MMPO2, MPO2, TNML_P, TNML_F, BosonMPS
from model.typeI import (
    CPDATypeI,
    CPDATypeI_GTN,
    LMPO2TypeI,
    LMPO2TypeI_GTN,
    MMPO2TypeI,
    MMPO2TypeI_GTN,
    MPO2TypeI,
    MPO2TypeI_GTN,
)
from model.image_models import CMPO2, CMPO3, CMPO2_GTN, CMPO3_GTN, BaselineCNN
from model.utils import (
    CLASSIFICATION_METRICS,
    REGRESSION_METRICS,
    compute_quality,
    create_inputs,
    create_inputs_tnml,
    encode_polynomial,
    encode_fourier,
)

import pandas as pd
import quimb

torch.set_default_dtype(torch.float64)
log = logging.getLogger(__name__)

quimb.tensor.set_contract_strategy("optimal")

# =============================================================================
# Run Tracking (loaded once for multirun efficiency)
# =============================================================================

# Global tracking DataFrame - loaded once at module import
# This avoids re-reading the CSV for each run in a Hydra multirun
_tracking_df = None


def get_tracking_df():
    """Get the tracking DataFrame, loading it if necessary."""
    global _tracking_df
    if _tracking_df is None:
        _tracking_df = load_tracking_file(DEFAULT_TRACKING_FILE)
    return _tracking_df


# =============================================================================
# Best Config Lookup
# =============================================================================

_best_configs_cache = {}


def load_best_config(trainer: str, model: str, dataset: str) -> dict | None:
    """Load best L/bond_dim for a model×dataset from conf/best_conf/{trainer}/{model}.yaml"""
    cache_key = (trainer, model)
    
    if cache_key not in _best_configs_cache:
        config_path = Path(__file__).parent / "conf" / "best_conf" / trainer / f"{model.lower()}.yaml"
        if not config_path.exists():
            _best_configs_cache[cache_key] = None
        else:
            with open(config_path) as f:
                data = yaml.safe_load(f)
            _best_configs_cache[cache_key] = data.get("_best_configs", {})
    
    configs = _best_configs_cache[cache_key]
    if configs is None:
        return None
    return configs.get(dataset)


# =============================================================================
# GPU Memory Utilities
# =============================================================================


def get_gpu_memory_info() -> dict:
    """Get GPU memory information in GB.

    Returns dict with:
        - total_gb: Total GPU memory
        - allocated_gb: Currently allocated memory
        - reserved_gb: Currently reserved by PyTorch
        - free_gb: Free memory available
        - peak_allocated_gb: Peak memory allocated during run
        - device_name: GPU device name
    """
    if not torch.cuda.is_available():
        return {"cuda_available": False}

    try:
        device = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(device)

        total = props.total_memory
        allocated = torch.cuda.memory_allocated(device)
        reserved = torch.cuda.memory_reserved(device)
        peak = torch.cuda.max_memory_allocated(device)

        # Convert to GB
        to_gb = lambda x: round(x / (1024**3), 3)

        return {
            "cuda_available": True,
            "device_name": props.name,
            "total_gb": to_gb(total),
            "allocated_gb": to_gb(allocated),
            "reserved_gb": to_gb(reserved),
            "free_gb": to_gb(total - reserved),
            "peak_allocated_gb": to_gb(peak),
        }
    except Exception as e:
        return {"cuda_available": True, "error": str(e)}


def reset_gpu_memory_stats():
    """Reset peak memory stats for accurate per-run tracking."""
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


# =============================================================================
# Model Registry
# =============================================================================

NTN_MODELS = {
    "MPO2": MPO2,
    "LMPO2": LMPO2,
    "MMPO2": MMPO2,
    "CPDA": CPDA,
    "MPO2TypeI": MPO2TypeI,
    "LMPO2TypeI": LMPO2TypeI,
    "MMPO2TypeI": MMPO2TypeI,
    "CPDATypeI": CPDATypeI,
    "TNML_P": TNML_P,
    "TNML_F": TNML_F,
}

GTN_TYPEI_MODELS = {
    "MPO2TypeI": MPO2TypeI_GTN,
    "LMPO2TypeI": LMPO2TypeI_GTN,
    "MMPO2TypeI": MMPO2TypeI_GTN,
    "CPDATypeI": CPDATypeI_GTN,
}

GTN_ONLY_MODELS = {
    "BosonMPS": BosonMPS,
}

IMAGE_MODELS = {
    "CMPO2": CMPO2,
    "CMPO3": CMPO3,
    "BaselineCNN": BaselineCNN,
}

IMAGE_GTN_MODELS = {
    "CMPO2": CMPO2_GTN,
    "CMPO3": CMPO3_GTN,
}


# =============================================================================
# Helper Functions
# =============================================================================


def get_reduced_dim(cfg: DictConfig, input_dim: int) -> int:
    """Calculate reduced_dim for LMPO2 models."""
    if cfg.model.get("reduced_dim"):
        return cfg.model.reduced_dim
    elif cfg.model.get("reduction_factor"):
        return max(2, int(input_dim * cfg.model.reduction_factor))
    else:
        return max(2, int(input_dim * 0.5))


def build_model_params(
    cfg: DictConfig,
    input_dim: int,
    output_dim: int,
    for_gtn: bool = False,
    raw_feature_count: int = None,
) -> dict:
    """Build model parameters from config."""
    is_typei = cfg.model.name.endswith("TypeI")
    is_tnml = cfg.model.name.startswith("TNML")

    params = {
        "phys_dim": raw_feature_count if is_tnml else input_dim,
        "output_dim": output_dim,
        "output_site": cfg.model.get("output_site"),
        "init_strength": cfg.model.get("init_strength", 0.001 if for_gtn else 0.1),
        "bond_dim": cfg.model.bond_dim,
        "L": cfg.model.L,
    }

    if is_typei:
        params["max_sites"] = params.pop("L")

    if "LMPO2" in cfg.model.name:
        params["reduced_dim"] = get_reduced_dim(cfg, input_dim)
        if not is_typei and not for_gtn:
            params["bond_dim_mpo"] = cfg.model.get("bond_dim_mpo", 2)

    return params


def create_gtn_loss(task: str, loss_fn_name: str | None = None) -> nn.Module:
    """Create loss function for GTN training."""
    if loss_fn_name:
        loss_map = {
            "mse": nn.MSELoss,
            "mae": nn.L1Loss,
            "huber": nn.HuberLoss,
            "cross_entropy": nn.CrossEntropyLoss,
        }
        if loss_fn_name not in loss_map:
            raise ValueError(f"Unknown loss function: {loss_fn_name}")
        return loss_map[loss_fn_name]()

    return nn.MSELoss() if task == "regression" else nn.CrossEntropyLoss()


def create_optimizer(
    name: str, parameters, lr: float, weight_decay: float
) -> optim.Optimizer:
    """Create optimizer by name."""
    optimizers = {
        "adam": lambda: optim.Adam(parameters, lr=lr, weight_decay=weight_decay),
        "adamw": lambda: optim.AdamW(parameters, lr=lr, weight_decay=weight_decay),
        "sgd": lambda: optim.SGD(
            parameters, lr=lr, weight_decay=weight_decay, momentum=0.9
        ),
    }
    if name not in optimizers:
        raise ValueError(f"Unknown optimizer: {name}")
    return optimizers[name]()


def create_model(
    cfg: DictConfig, input_dim: int, output_dim: int, raw_feature_count: int = None
):
    """Create model instance from config."""
    if cfg.model.name in GTN_ONLY_MODELS:
        return None

    model_cls = NTN_MODELS.get(cfg.model.name)
    if model_cls is None:
        raise ValueError(
            f"Unknown model: {cfg.model.name}. Available: {list(NTN_MODELS.keys())}"
        )

    params = build_model_params(
        cfg, input_dim, output_dim, for_gtn=False, raw_feature_count=raw_feature_count
    )
    return model_cls(**params)


def run_ntn(cfg: DictConfig, model, data: dict, output_dir: Path) -> dict:
    """Run Newton-based training."""
    is_typei = cfg.model.name.endswith("TypeI")
    task = cfg.dataset.task

    # Setup loss and metrics
    if task == "regression":
        loss_fn = MSELoss()
        eval_metrics = REGRESSION_METRICS
    else:
        loss_fn = CrossEntropyLoss()
        eval_metrics = CLASSIFICATION_METRICS

    # Ridge schedule (jitter for NTN)
    n_epochs = cfg.trainer.n_epochs
    ridge_schedule = [
        max(cfg.trainer.ridge * (cfg.trainer.ridge_decay**epoch), cfg.trainer.ridge_min)
        for epoch in range(n_epochs)
    ]

    # Move model to device
    if is_typei:
        for tn in model.tns:
            move_tn_to_device(tn)
    else:
        move_tn_to_device(model.tn)

    # Reset GPU memory stats for accurate peak tracking
    reset_gpu_memory_stats()
    gpu_info_before = get_gpu_memory_info()

    if is_typei:
        ntn_kwargs = {
            "tns": model.tns,
            "input_dims_list": model.input_dims_list,
            "input_labels_list": model.input_labels_list,
            "output_dims": model.output_dims,
            "loss": loss_fn,
            "X_train": data["X_train"],
            "y_train": data["y_train"],
            "X_val": data["X_val"],
            "y_val": data["y_val"],
            "batch_size": cfg.dataset.batch_size,
            "not_trainable_tags": getattr(model, "not_trainable_tags", None),
        }
        if cfg.trainer.get("evaluate_test", False):
            ntn_kwargs["X_test"] = data["X_test"]
            ntn_kwargs["y_test"] = data["y_test"]
        ntn = NTN_Ensemble(**ntn_kwargs)
        loader_val = ntn.val_data
    else:
        encoding = getattr(model, "encoding", None)
        poly_degree = (
            getattr(model, "poly_degree", None) if encoding == "polynomial" else None
        )
        loader_train = create_inputs(
            X=data["X_train"],
            y=data["y_train"],
            input_labels=model.input_labels,
            output_labels=model.output_dims,
            batch_size=cfg.dataset.batch_size,
            append_bias=(encoding is None),
            encoding=encoding,
            poly_degree=poly_degree,
        )
        loader_val = create_inputs(
            X=data["X_val"],
            y=data["y_val"],
            input_labels=model.input_labels,
            output_labels=model.output_dims,
            batch_size=cfg.dataset.batch_size,
            append_bias=(encoding is None),
            encoding=encoding,
            poly_degree=poly_degree,
        )
        ntn = NTN(
            tn=model.tn,
            output_dims=model.output_dims,
            input_dims=model.input_dims,
            loss=loss_fn,
            data_stream=loader_train,
        )

    evaluate_test = cfg.trainer.get("evaluate_test", False)
    loader_test = None

    if evaluate_test and not is_typei:
        loader_test = create_inputs(
            X=data["X_test"],
            y=data["y_test"],
            input_labels=model.input_labels,
            output_labels=model.output_dims,
            batch_size=cfg.dataset.batch_size,
            append_bias=(encoding is None),
            encoding=encoding,
            poly_degree=poly_degree,
        )

    metrics_log = []
    train_start_time = time.time()
    last_callback_time = train_start_time

    def callback_epoch(epoch, scores_train, scores_val, info):
        nonlocal last_callback_time
        current_time = time.time()
        wall_time = current_time - train_start_time
        epoch_time = current_time - last_callback_time
        last_callback_time = current_time

        metrics = {
            "epoch": epoch,
            "train_loss": float(scores_train["loss"]),
            "train_quality": float(compute_quality(scores_train)),
            "val_loss": float(scores_val["loss"]),
            "val_quality": float(compute_quality(scores_val)),
            "ridge": float(info["jitter"]),
            "epoch_time": float(epoch_time),
            "wall_time": float(wall_time),
        }
        if evaluate_test:
            if is_typei:
                scores_test = ntn.evaluate(eval_metrics, split="test")
            elif loader_test is not None:
                scores_test = ntn.evaluate(eval_metrics, data_stream=loader_test)
            else:
                scores_test = None
            if scores_test is not None:
                metrics["test_loss"] = float(scores_test["loss"])
                metrics["test_quality"] = float(compute_quality(scores_test))
        metrics_log.append(metrics)

    oom_error = False
    try:
        scores_train, scores_val = ntn.fit(
            n_epochs=n_epochs,
            regularize=True,
            jitter=ridge_schedule,
            eval_metrics=eval_metrics,
            val_data=loader_val,
            verbose=True,
            callback_epoch=callback_epoch,
            adaptive_jitter=cfg.trainer.adaptive_ridge,
            patience=cfg.trainer.patience,
            min_delta=cfg.trainer.min_delta,
            train_selection=cfg.trainer.train_selection,
        )
        success = True
        singular = ntn.singular_encountered

    except SingularMatrixError:
        success = False
        singular = True
        # Will extract best values from metrics_log below
        scores_train = None
        scores_val = None
    except torch.OutOfMemoryError as e:
        success = False
        singular = False
        oom_error = True
        log.error(f"CUDA out of memory: {e}")
        scores_train = None
        scores_val = None

    best_epoch = -1
    best_train_loss = float("inf")
    best_train_quality = None
    best_val_loss = float("inf")
    best_val_quality = None
    best_test_loss = None
    best_test_quality = None

    if metrics_log:
        best_val_q = float("-inf")
        for m in metrics_log:
            if m["val_quality"] is not None and m["val_quality"] > best_val_q:
                best_val_q = m["val_quality"]
                best_epoch = m["epoch"]
                best_train_loss = m["train_loss"]
                best_train_quality = m["train_quality"]
                best_val_loss = m["val_loss"]
                best_val_quality = m["val_quality"]
                best_test_loss = m.get("test_loss")
                best_test_quality = m.get("test_quality")

    if success and scores_train is not None:
        best_train_loss = float(scores_train["loss"])
        best_train_quality = float(compute_quality(scores_train))
        best_val_loss = float(scores_val["loss"])
        best_val_quality = float(compute_quality(scores_val))

    total_time = time.time() - train_start_time
    gpu_info_after = get_gpu_memory_info()

    result = {
        "success": success,
        "singular": singular,
        "oom_error": oom_error,
        "train_loss": best_train_loss,
        "train_quality": best_train_quality,
        "val_loss": best_val_loss,
        "val_quality": best_val_quality,
        "best_epoch": best_epoch,
        "metrics_log": metrics_log,
        "total_time": float(total_time),
        "gpu_memory": {
            "before": gpu_info_before,
            "after": gpu_info_after,
        },
    }

    if evaluate_test:
        result["test_loss"] = best_test_loss
        result["test_quality"] = best_test_quality

    result["_ntn"] = ntn
    result["_loader_val"] = loader_val
    result["_loader_test"] = loader_test
    result["_is_typei"] = is_typei

    return result


def run_dmrg(cfg: DictConfig, model, data: dict, output_dir: Path) -> dict:
    """Run 2-site DMRG training for TNML models."""
    task = cfg.dataset.task
    
    if task == "regression":
        loss_fn = MSELoss()
        eval_metrics = REGRESSION_METRICS
    else:
        loss_fn = CrossEntropyLoss()
        eval_metrics = CLASSIFICATION_METRICS
    
    n_epochs = cfg.trainer.n_epochs
    ridge_schedule = [
        max(cfg.trainer.ridge * (cfg.trainer.ridge_decay**epoch), cfg.trainer.ridge_min)
        for epoch in range(n_epochs)
    ]
    
    move_tn_to_device(model.tn)
    reset_gpu_memory_stats()
    gpu_info_before = get_gpu_memory_info()
    
    encoding = getattr(model, "encoding", None)
    poly_degree = getattr(model, "poly_degree", None) if encoding == "polynomial" else None
    
    loader_train = create_inputs(
        X=data["X_train"],
        y=data["y_train"],
        input_labels=model.input_labels,
        output_labels=model.output_dims,
        batch_size=cfg.dataset.batch_size,
        append_bias=False,
        encoding=encoding,
        poly_degree=poly_degree,
    )
    loader_val = create_inputs(
        X=data["X_val"],
        y=data["y_val"],
        input_labels=model.input_labels,
        output_labels=model.output_dims,
        batch_size=cfg.dataset.batch_size,
        append_bias=False,
        encoding=encoding,
        poly_degree=poly_degree,
    )
    
    dmrg = DMRG(
        tn=model.tn,
        output_dims=model.output_dims,
        input_dims=model.input_dims,
        loss=loss_fn,
        data_stream=loader_train,
    )
    
    max_bond = cfg.trainer.get("max_bond", None)
    if max_bond is None:
        max_bond=model.bond_dim
    cutoff = cfg.trainer.get("cutoff", 1e-10)
    
    metrics_log = []
    train_start_time = time.time()
    last_callback_time = train_start_time
    
    def callback_epoch(epoch, scores_train, scores_val, info):
        nonlocal last_callback_time
        current_time = time.time()
        wall_time = current_time - train_start_time
        epoch_time = current_time - last_callback_time
        last_callback_time = current_time
        
        metrics = {
            "epoch": epoch,
            "train_loss": float(scores_train["loss"]),
            "train_quality": float(compute_quality(scores_train)),
            "val_loss": float(scores_val["loss"]),
            "val_quality": float(compute_quality(scores_val)),
            "ridge": float(info["jitter"]),
            "epoch_time": float(epoch_time),
            "wall_time": float(wall_time),
        }
        metrics_log.append(metrics)
    
    oom_error = False
    try:
        scores_train, scores_val = dmrg.fit(
            n_epochs=n_epochs,
            regularize=True,
            jitter=ridge_schedule,
            verbose=True,
            eval_metrics=eval_metrics,
            val_data=loader_val,
            callback_epoch=callback_epoch,
            patience=cfg.trainer.patience,
            min_delta=cfg.trainer.min_delta,
            max_bond=max_bond,
            cutoff=cutoff,
        )
        success = True
        singular = dmrg.singular_encountered
    except SingularMatrixError:
        success = False
        singular = True
        scores_train = None
        scores_val = None
    except torch.OutOfMemoryError as e:
        success = False
        singular = False
        oom_error = True
        log.error(f"CUDA out of memory: {e}")
        scores_train = None
        scores_val = None
    
    best_epoch = -1
    best_train_loss = float("inf")
    best_train_quality = None
    best_val_loss = float("inf")
    best_val_quality = None
    
    if metrics_log:
        best_val_q = float("-inf")
        for m in metrics_log:
            if m["val_quality"] is not None and m["val_quality"] > best_val_q:
                best_val_q = m["val_quality"]
                best_epoch = m["epoch"]
                best_train_loss = m["train_loss"]
                best_train_quality = m["train_quality"]
                best_val_loss = m["val_loss"]
                best_val_quality = m["val_quality"]
    
    if success and scores_train is not None:
        best_train_loss = float(scores_train["loss"])
        best_train_quality = float(compute_quality(scores_train))
        best_val_loss = float(scores_val["loss"])
        best_val_quality = float(compute_quality(scores_val))
    
    total_time = time.time() - train_start_time
    gpu_info_after = get_gpu_memory_info()
    
    return {
        "success": success,
        "singular": singular,
        "oom_error": oom_error,
        "train_loss": best_train_loss,
        "train_quality": best_train_quality,
        "val_loss": best_val_loss,
        "val_quality": best_val_quality,
        "best_epoch": best_epoch,
        "metrics_log": metrics_log,
        "total_time": float(total_time),
        "gpu_memory": {"before": gpu_info_before, "after": gpu_info_after},
        "_dmrg": dmrg,
        "_loader_val": loader_val,
    }


def run_gtn(cfg: DictConfig, model, data: dict, output_dir: Path) -> dict:
    """Run gradient-based training."""
    task = cfg.dataset.task
    is_typei = cfg.model.name.endswith("TypeI")

    # Setup
    criterion = create_gtn_loss(task, cfg.trainer.get("loss_fn"))
    n_epochs = cfg.trainer.n_epochs
    batch_size = cfg.dataset.batch_size
    patience = cfg.trainer.get("patience")
    min_delta = cfg.trainer.get("min_delta", 0.0)

    # Create GTN model
    input_dim = data["X_train"].shape[1] + 1  # +1 for bias term
    output_dim = data["y_train"].shape[1] if data["y_train"].ndim > 1 else 1

    is_gtn_only = cfg.model.name in GTN_ONLY_MODELS

    if is_gtn_only:
        gtn_cls = GTN_ONLY_MODELS[cfg.model.name]
        params = build_model_params(cfg, input_dim, output_dim, for_gtn=True)
        gtn_model = gtn_cls(**params)
    elif is_typei:
        gtn_cls = GTN_TYPEI_MODELS.get(cfg.model.name)
        if gtn_cls is None:
            raise ValueError(f"Unknown TypeI model for GTN: {cfg.model.name}")
        params = build_model_params(cfg, input_dim, output_dim, for_gtn=True)
        gtn_model = gtn_cls(**params)
    else:
        gtn_model = GTN(
            tn=model.tn, output_dims=model.output_dims, input_dims=model.input_dims
        )

    gtn_model = gtn_model.to(DEVICE)

    # Create optimizer (weight_decay = 2 * ridge to match NTN's ridge behavior)
    weight_decay = 2 * cfg.trainer.ridge
    optimizer = create_optimizer(
        cfg.trainer.get("optimizer", "adam").lower(),
        gtn_model.parameters(),
        cfg.trainer.lr,
        weight_decay,
    )

    encoding = getattr(model, "encoding", None) if model is not None else None
    is_tnml = encoding is not None

    if is_tnml:
        poly_degree = getattr(model, "poly_degree", None)
        n_features = data["X_train"].shape[1]
        if encoding == "polynomial":
            X_train_enc = encode_polynomial(data["X_train"], poly_degree)
            X_val_enc = encode_polynomial(data["X_val"], poly_degree)
        else:
            X_train_enc = encode_fourier(data["X_train"])
            X_val_enc = encode_fourier(data["X_val"])
    else:
        X_train_enc = torch.cat(
            [
                data["X_train"],
                torch.ones(
                    data["X_train"].shape[0],
                    1,
                    dtype=data["X_train"].dtype,
                    device=data["X_train"].device,
                ),
            ],
            dim=1,
        )
        X_val_enc = torch.cat(
            [
                data["X_val"],
                torch.ones(
                    data["X_val"].shape[0],
                    1,
                    dtype=data["X_val"].dtype,
                    device=data["X_val"].device,
                ),
            ],
            dim=1,
        )

    evaluate_test = cfg.trainer.get("evaluate_test", False)

    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X_train_enc, data["y_train"]),
        batch_size=batch_size,
        shuffle=True,
    )
    val_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X_val_enc, data["y_val"]),
        batch_size=batch_size,
        shuffle=False,
    )

    test_loader = None
    if evaluate_test:
        if is_tnml:
            if encoding == "polynomial":
                X_test_enc = encode_polynomial(data["X_test"], poly_degree)
            else:
                X_test_enc = encode_fourier(data["X_test"])
        else:
            X_test_enc = torch.cat(
                [
                    data["X_test"],
                    torch.ones(
                        data["X_test"].shape[0],
                        1,
                        dtype=data["X_test"].dtype,
                        device=data["X_test"].device,
                    ),
                ],
                dim=1,
            )
        test_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(X_test_enc, data["y_test"]),
            batch_size=batch_size,
            shuffle=False,
        )

    def prepare_input(batch_data):
        if is_tnml:
            return [batch_data[:, i, :] for i in range(batch_data.shape[1])]
        return batch_data

    def evaluate(loader):
        gtn_model.eval()
        total_loss = 0
        all_preds, all_targets = [], []

        with torch.no_grad():
            for batch_data, batch_target in loader:
                batch_data, batch_target = (
                    batch_data.to(DEVICE),
                    batch_target.to(DEVICE),
                )
                output = gtn_model(prepare_input(batch_data))
                loss = criterion(output, batch_target)
                total_loss += loss.item() * batch_data.size(0)
                all_preds.append(output.cpu())
                all_targets.append(batch_target.cpu())

        preds = torch.cat(all_preds, dim=0)
        targets = torch.cat(all_targets, dim=0)
        avg_loss = total_loss / len(loader.dataset)

        if task == "regression":
            ss_res = torch.sum((targets - preds) ** 2).item()
            ss_tot = torch.sum((targets - targets.mean()) ** 2).item()
            quality = 1 - ss_res / ss_tot if ss_tot > 0 else float("-inf")
        else:
            pred_labels = preds.argmax(dim=1)
            target_labels = targets.argmax(dim=1)
            quality = (pred_labels == target_labels).float().mean().item()

        return avg_loss, quality

    metrics_log = []
    best_val_quality = float("-inf")
    best_train_quality = float("-inf")
    best_epoch = -1
    patience_counter = 0
    train_selection = cfg.trainer.get("train_selection", False)
    train_start_time = time.time()

    init_train_loss, init_train_quality = evaluate(train_loader)
    init_val_loss, init_val_quality = evaluate(val_loader)
    init_metrics = {
        "epoch": -1,
        "train_loss": float(init_train_loss),
        "train_quality": float(init_train_quality),
        "val_loss": float(init_val_loss),
        "val_quality": float(init_val_quality),
        "epoch_time": 0.0,
        "wall_time": 0.0,
    }
    if evaluate_test:
        init_test_loss, init_test_quality = evaluate(test_loader)
        init_metrics["test_loss"] = float(init_test_loss)
        init_metrics["test_quality"] = float(init_test_quality)
    metrics_log.append(init_metrics)

    for epoch in range(n_epochs):
        epoch_start_time = time.time()
        gtn_model.train()
        train_loss = 0.0

        for batch_data, batch_target in train_loader:
            batch_data, batch_target = batch_data.to(DEVICE), batch_target.to(DEVICE)
            optimizer.zero_grad()
            output = gtn_model(prepare_input(batch_data))
            loss = criterion(output, batch_target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * batch_data.size(0)

        train_loss /= len(train_loader.dataset)
        _, train_quality = evaluate(train_loader)
        val_loss, val_quality = evaluate(val_loader)

        wall_time = time.time() - train_start_time
        epoch_time = time.time() - epoch_start_time

        epoch_metrics = {
            "epoch": epoch,
            "train_loss": float(train_loss),
            "train_quality": float(train_quality),
            "val_loss": float(val_loss),
            "val_quality": float(val_quality),
            "epoch_time": float(epoch_time),
            "wall_time": float(wall_time),
        }

        if evaluate_test:
            test_loss, test_quality = evaluate(test_loader)
            epoch_metrics["test_loss"] = float(test_loss)
            epoch_metrics["test_quality"] = float(test_quality)

        metrics_log.append(epoch_metrics)

        # Model selection (same logic as NTN)
        is_best = False
        if train_selection:
            # Use training quality as tiebreaker when val quality is equal
            val_improved = val_quality > best_val_quality + min_delta
            train_improved = train_quality > best_train_quality + min_delta
            val_same = abs(val_quality - best_val_quality) < min_delta

            if val_improved or (val_same and train_improved):
                best_val_quality = val_quality
                best_train_quality = train_quality
                best_epoch = epoch
                is_best = True
                patience_counter = 0
            else:
                patience_counter += 1
        else:
            # Only consider validation quality
            if val_quality > best_val_quality + min_delta:
                best_val_quality = val_quality
                best_train_quality = train_quality
                best_epoch = epoch
                is_best = True
                patience_counter = 0
            else:
                patience_counter += 1

        if patience is not None and patience_counter >= patience:
            log.info(
                f"Early stopping at epoch {epoch + 1} (no improvement for {patience} epochs)"
            )
            break

    # Get best metrics from metrics_log (consistent with NTN behavior)
    if best_epoch >= 0 and metrics_log:
        best_metrics = metrics_log[best_epoch]
        best_train_loss = best_metrics["train_loss"]
        best_train_quality = best_metrics["train_quality"]
        best_val_loss = best_metrics["val_loss"]
        best_val_quality = best_metrics["val_quality"]
        best_test_loss = best_metrics.get("test_loss")
        best_test_quality = best_metrics.get("test_quality")
    else:
        best_train_loss, best_train_quality = evaluate(train_loader)
        best_val_loss, best_val_quality = evaluate(val_loader)
        if evaluate_test:
            best_test_loss, best_test_quality = evaluate(test_loader)
        else:
            best_test_loss, best_test_quality = None, None

    total_time = time.time() - train_start_time

    result = {
        "success": True,
        "singular": False,
        "train_loss": float(best_train_loss),
        "train_quality": float(best_train_quality),
        "val_loss": float(best_val_loss),
        "val_quality": float(best_val_quality),
        "best_epoch": best_epoch,
        "metrics_log": metrics_log,
        "total_time": float(total_time),
    }

    if evaluate_test:
        result["test_loss"] = float(best_test_loss) if best_test_loss is not None else None
        result["test_quality"] = float(best_test_quality) if best_test_quality is not None else None

    result["_gtn_model"] = gtn_model
    result["_val_loader"] = val_loader
    result["_test_loader"] = test_loader
    result["_prepare_input"] = prepare_input

    return result


def is_image_model(model_name: str) -> bool:
    return model_name in IMAGE_MODELS


def create_image_model(cfg: DictConfig, dataset_info: dict):
    model_name = cfg.model.name
    if model_name == "CMPO2":
        return CMPO2(
            L=cfg.model.L,
            pixel_dim=dataset_info["pixels_per_patch"],
            patch_dim=dataset_info["n_patches"],
            pixel_bond_dim=cfg.model.rank_pixel,
            patch_bond_dim=cfg.model.rank_patch,
            output_dim=dataset_info["n_classes"],
            init_strength=cfg.model.get("init_strength", 0.01),
        )
    elif model_name == "CMPO3":
        return CMPO3(
            L=cfg.model.L,
            channel_dim=dataset_info["n_channels"],
            pixel_dim=dataset_info["pixels_per_patch"],
            patch_dim=dataset_info["n_patches"],
            channel_bond_dim=cfg.model.rank_channel,
            pixel_bond_dim=cfg.model.rank_pixel,
            patch_bond_dim=cfg.model.rank_patch,
            output_dim=dataset_info["n_classes"],
            init_strength=cfg.model.get("init_strength", 0.01),
        )
    elif model_name == "BaselineCNN":
        return BaselineCNN(
            input_channels=dataset_info["channels"],
            image_size=dataset_info["image_size"],
            n_classes=dataset_info["n_classes"],
            n_conv_layers=cfg.model.n_conv_layers,
            base_channels=cfg.model.base_channels,
            fc_hidden_dim=cfg.model.fc_hidden_dim,
            kernel_size=cfg.model.get("kernel_size", 3),
            use_batchnorm=cfg.model.get("use_batchnorm", True),
        )
    else:
        raise ValueError(f"Unknown image model: {model_name}")


def create_inputs_image(X, y, input_labels, output_labels, batch_size):
    from model.builder import Inputs
    return Inputs(
        inputs=[X],
        outputs=[y],
        outputs_labels=output_labels,
        input_labels=input_labels,
        batch_dim="s",
        batch_size=batch_size,
    )


def run_ntn_image(cfg: DictConfig, model, data: dict, output_dir: Path) -> dict:
    move_tn_to_device(model.tn)

    n_epochs = cfg.trainer.n_epochs
    batch_size = cfg.dataset.batch_size

    ridge_schedule = [
        max(cfg.trainer.ridge * (cfg.trainer.ridge_decay**epoch), cfg.trainer.ridge_min)
        for epoch in range(n_epochs)
    ]

    reset_gpu_memory_stats()
    gpu_info_before = get_gpu_memory_info()

    loader_train = create_inputs_image(
        data["X_train"], data["y_train"],
        model.input_labels, model.output_dims, batch_size,
    )
    loader_val = create_inputs_image(
        data["X_val"], data["y_val"],
        model.input_labels, model.output_dims, batch_size,
    )
    loader_test = create_inputs_image(
        data["X_test"], data["y_test"],
        model.input_labels, model.output_dims, batch_size,
    )

    loss_fn = CrossEntropyLoss()
    ntn = NTN(
        tn=model.tn,
        output_dims=model.output_dims,
        input_dims=model.input_dims,
        loss=loss_fn,
        data_stream=loader_train,
    )

    trainable_nodes = ntn._get_trainable_nodes()
    groups = defaultdict(list)
    for node in trainable_nodes:
        groups[int(node.split('_')[0])].append(node)
    indices = sorted(groups.keys())
    sequence = indices + indices[-2:0:-1]
    ordered_list = [node for i in sequence for node in groups[i]]

    metrics_log = []
    train_start_time = time.time()
    last_callback_time = train_start_time

    def callback_epoch(epoch, scores_train, scores_val, info):
        nonlocal last_callback_time
        current_time = time.time()
        wall_time = current_time - train_start_time
        epoch_time = current_time - last_callback_time
        last_callback_time = current_time

        metrics = {
            "epoch": epoch,
            "train_loss": float(scores_train["loss"]),
            "train_quality": float(compute_quality(scores_train)),
            "val_loss": float(scores_val["loss"]),
            "val_quality": float(compute_quality(scores_val)),
            "ridge": float(info["jitter"]),
            "epoch_time": float(epoch_time),
            "wall_time": float(wall_time),
        }
        scores_test = ntn.evaluate(CLASSIFICATION_METRICS, data_stream=loader_test)
        metrics["test_loss"] = float(scores_test["loss"])
        metrics["test_quality"] = float(compute_quality(scores_test))
        metrics_log.append(metrics)

    oom_error = False
    try:
        scores_train, scores_val = ntn.fit(
            n_epochs=n_epochs,
            regularize=True,
            jitter=ridge_schedule,
            eval_metrics=CLASSIFICATION_METRICS,
            val_data=loader_val,
            verbose=True,
            callback_epoch=callback_epoch,
            adaptive_jitter=cfg.trainer.adaptive_ridge,
            patience=cfg.trainer.patience,
            min_delta=cfg.trainer.min_delta,
            train_selection=cfg.trainer.train_selection,
            full_sweep_order=ordered_list,
        )
        success = True
        singular = ntn.singular_encountered

    except SingularMatrixError:
        success = False
        singular = True
        scores_train = None
        scores_val = None
    except torch.OutOfMemoryError as e:
        success = False
        singular = False
        oom_error = True
        log.error(f"CUDA out of memory: {e}")
        scores_train = None
        scores_val = None

    best_epoch = -1
    best_train_loss = float("inf")
    best_train_quality = None
    best_val_loss = float("inf")
    best_val_quality = None
    best_test_loss = None
    best_test_quality = None

    if metrics_log:
        best_val_q = float("-inf")
        for m in metrics_log:
            if m["val_quality"] is not None and m["val_quality"] > best_val_q:
                best_val_q = m["val_quality"]
                best_epoch = m["epoch"]
                best_train_loss = m["train_loss"]
                best_train_quality = m["train_quality"]
                best_val_loss = m["val_loss"]
                best_val_quality = m["val_quality"]
                best_test_loss = m.get("test_loss")
                best_test_quality = m.get("test_quality")

    if success and scores_train is not None:
        best_train_loss = float(scores_train["loss"])
        best_train_quality = float(compute_quality(scores_train))
        best_val_loss = float(scores_val["loss"])
        best_val_quality = float(compute_quality(scores_val))

    total_time = time.time() - train_start_time
    gpu_info_after = get_gpu_memory_info()

    return {
        "success": success,
        "singular": singular,
        "oom_error": oom_error,
        "train_loss": best_train_loss,
        "train_quality": best_train_quality,
        "val_loss": best_val_loss,
        "val_quality": best_val_quality,
        "test_loss": best_test_loss,
        "test_quality": best_test_quality,
        "best_epoch": best_epoch,
        "metrics_log": metrics_log,
        "total_time": float(total_time),
        "gpu_memory": {
            "before": gpu_info_before,
            "after": gpu_info_after,
        },
    }


def run_gtn_image(cfg: DictConfig, model, data: dict, output_dir: Path) -> dict:
    gtn_cls = IMAGE_GTN_MODELS[cfg.model.name]
    gtn_model = gtn_cls(model).to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = create_optimizer(
        cfg.trainer.get("optimizer", "adamw").lower(),
        gtn_model.parameters(),
        cfg.trainer.lr,
        cfg.trainer.get("weight_decay", 0.01),
    )

    batch_size = cfg.dataset.batch_size
    n_epochs = cfg.trainer.n_epochs
    patience = cfg.trainer.patience
    min_delta = cfg.trainer.min_delta

    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(data["X_train"], data["y_train"]),
        batch_size=batch_size, shuffle=True,
    )
    val_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(data["X_val"], data["y_val"]),
        batch_size=batch_size, shuffle=False,
    )
    test_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(data["X_test"], data["y_test"]),
        batch_size=batch_size, shuffle=False,
    )

    def evaluate_img(loader):
        gtn_model.eval()
        total_loss, correct, total = 0, 0, 0
        with torch.no_grad():
            for batch_data, batch_target in loader:
                output = gtn_model(batch_data)
                loss = criterion(output, batch_target)
                total_loss += loss.item() * batch_data.size(0)
                pred = output.argmax(dim=1)
                target_labels = batch_target.argmax(dim=1)
                correct += (pred == target_labels).sum().item()
                total += batch_target.size(0)
        return total_loss / len(loader.dataset), correct / total if total > 0 else 0.0

    metrics_log = []
    best_val_quality = 0.0
    best_epoch = -1
    patience_counter = 0
    train_start_time = time.time()

    init_train_loss, init_train_quality = evaluate_img(train_loader)
    init_val_loss, init_val_quality = evaluate_img(val_loader)
    init_test_loss, init_test_quality = evaluate_img(test_loader)
    metrics_log.append({
        "epoch": -1,
        "train_loss": float(init_train_loss),
        "train_quality": float(init_train_quality),
        "val_loss": float(init_val_loss),
        "val_quality": float(init_val_quality),
        "test_loss": float(init_test_loss),
        "test_quality": float(init_test_quality),
        "epoch_time": 0.0,
        "wall_time": 0.0,
    })

    for epoch in range(n_epochs):
        epoch_start_time = time.time()
        gtn_model.train()
        train_loss = 0.0

        for batch_data, batch_target in train_loader:
            optimizer.zero_grad()
            output = gtn_model(batch_data)
            loss = criterion(output, batch_target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * batch_data.size(0)

        train_loss /= len(train_loader.dataset)
        _, train_quality = evaluate_img(train_loader)
        val_loss, val_quality = evaluate_img(val_loader)

        wall_time = time.time() - train_start_time
        epoch_time = time.time() - epoch_start_time

        metrics_log.append({
            "epoch": epoch,
            "train_loss": float(train_loss),
            "train_quality": float(train_quality),
            "val_loss": float(val_loss),
            "val_quality": float(val_quality),
            "epoch_time": float(epoch_time),
            "wall_time": float(wall_time),
        })

        if val_quality > best_val_quality + min_delta:
            best_val_quality = val_quality
            best_epoch = epoch
            patience_counter = 0
        else:
            patience_counter += 1

        if patience is not None and patience_counter >= patience:
            log.info(f"Early stopping at epoch {epoch + 1}")
            break

        if epoch % 10 == 0 or epoch == n_epochs - 1:
            log.info(f"Epoch {epoch + 1:3d} | Train: {train_quality:.4f} | Val: {val_quality:.4f} | Time: {wall_time:.1f}s")

    total_time = time.time() - train_start_time
    test_loss, test_quality = evaluate_img(test_loader)

    return {
        "success": True,
        "singular": False,
        "train_quality": float(train_quality),
        "val_quality": float(best_val_quality),
        "test_quality": float(test_quality),
        "best_epoch": best_epoch,
        "metrics_log": metrics_log,
        "total_time": float(total_time),
    }


def run_cnn(cfg: DictConfig, model: nn.Module, data: dict, output_dir: Path) -> dict:
    model = model.float().to(DEVICE)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = create_optimizer(
        cfg.trainer.get("optimizer", "adamw").lower(),
        model.parameters(),
        cfg.trainer.lr,
        cfg.trainer.get("weight_decay", 0.01),
    )

    batch_size = cfg.dataset.batch_size
    n_epochs = cfg.trainer.n_epochs
    patience = cfg.trainer.patience
    min_delta = cfg.trainer.min_delta

    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(data["X_train"], data["y_train"]),
        batch_size=batch_size, shuffle=True,
    )
    val_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(data["X_val"], data["y_val"]),
        batch_size=batch_size, shuffle=False,
    )
    test_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(data["X_test"], data["y_test"]),
        batch_size=batch_size, shuffle=False,
    )

    def evaluate_cnn(loader):
        model.eval()
        total_loss, correct, total = 0, 0, 0
        with torch.no_grad():
            for batch_data, batch_target in loader:
                batch_data = batch_data.to(DEVICE)
                batch_target = batch_target.to(DEVICE)
                output = model(batch_data)
                loss = criterion(output, batch_target)
                total_loss += loss.item() * batch_data.size(0)
                pred = output.argmax(dim=1)
                correct += (pred == batch_target).sum().item()
                total += batch_target.size(0)
        return total_loss / len(loader.dataset), correct / total if total > 0 else 0.0

    metrics_log = []
    best_val_quality = 0.0
    best_epoch = -1
    patience_counter = 0
    best_model_state = None
    train_start_time = time.time()

    for epoch in range(n_epochs):
        epoch_start_time = time.time()
        model.train()
        train_loss = 0.0

        for batch_data, batch_target in train_loader:
            batch_data = batch_data.to(DEVICE)
            batch_target = batch_target.to(DEVICE)
            optimizer.zero_grad()
            output = model(batch_data)
            loss = criterion(output, batch_target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * batch_data.size(0)

        train_loss /= len(train_loader.dataset)
        _, train_quality = evaluate_cnn(train_loader)
        val_loss, val_quality = evaluate_cnn(val_loader)

        wall_time = time.time() - train_start_time
        epoch_time = time.time() - epoch_start_time

        metrics_log.append({
            "epoch": epoch,
            "train_loss": float(train_loss),
            "train_quality": float(train_quality),
            "val_loss": float(val_loss),
            "val_quality": float(val_quality),
            "epoch_time": float(epoch_time),
            "wall_time": float(wall_time),
        })

        if val_quality > best_val_quality + min_delta:
            best_val_quality = val_quality
            best_epoch = epoch
            patience_counter = 0
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1

        if patience is not None and patience_counter >= patience:
            log.info(f"Early stopping at epoch {epoch + 1}")
            break

        if epoch % 10 == 0 or epoch == n_epochs - 1:
            log.info(f"Epoch {epoch + 1:3d} | Train: {train_quality:.4f} | Val: {val_quality:.4f} | Time: {wall_time:.1f}s")

    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        model.to(DEVICE)

    total_time = time.time() - train_start_time
    test_loss, test_quality = evaluate_cnn(test_loader)

    return {
        "success": True,
        "singular": False,
        "train_quality": float(train_quality),
        "val_quality": float(best_val_quality),
        "test_quality": float(test_quality),
        "best_epoch": best_epoch,
        "metrics_log": metrics_log,
        "total_time": float(total_time),
    }


def _format_run_summary(cfg: DictConfig, output_dir: Path) -> str:
    model = cfg.model.name
    dataset = cfg.dataset.name
    trainer = cfg.trainer.type.upper()
    seed = cfg.seed
    if model == "BaselineCNN":
        nlayers = cfg.model.n_conv_layers
        ch = cfg.model.base_channels
        fc = cfg.model.fc_hidden_dim
        return f"{model}/{dataset}/{trainer} nlayers={nlayers} ch={ch} fc={fc} seed={seed} -> {output_dir}"
    L = cfg.model.get("L", "?")
    bd = cfg.model.get("bond_dim", cfg.model.get("rank_pixel", "?"))
    rf = cfg.model.get("reduction_factor", "")
    rf_str = f"_rf{rf}" if rf else ""
    return f"{model}/{dataset}/{trainer} L={L} bd={bd}{rf_str} seed={seed} -> {output_dir}"


def _main_impl(cfg: DictConfig):
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    if cfg.trainer.get("evaluate_test", False):
        best = load_best_config(cfg.trainer.type, cfg.model.name, cfg.dataset.name)
        if best:
            cfg.model.L = best["L"]
            cfg.model.bond_dim = best["bond_dim"]
            log.info(f"Using best config for {cfg.model.name}/{cfg.dataset.name}: L={best['L']}, bond_dim={best['bond_dim']}")
        else:
            log.warning(f"No best config found for {cfg.trainer.type}/{cfg.model.name}/{cfg.dataset.name}, using provided config")

    output_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    run_summary = _format_run_summary(cfg, output_dir)

    # Check for completed run (skip if already successful)
    skip_completed = cfg.get("skip_completed", True)

    if skip_completed:
        # First check tracking CSV (fast, no filesystem access to outputs)
        run_id = generate_run_id(cfg)
        tracking_df = get_tracking_df()
        skip, reason, cached_val = should_skip_run(tracking_df, run_id)

        if skip:
            val_str = f"val={cached_val:.4f}" if cached_val is not None else ""
            log.info(f"⏭ SKIP {run_summary} | {reason} {val_str}")
            return cached_val if cached_val is not None else float("-inf")
        elif cached_val is not None:
            # Not skipping but we have previous info (e.g., OOM retry)
            log.info(f"♻ RETRY {run_summary} | {reason}")

        # Fallback: check results.json for backwards compatibility
        # (handles runs completed before tracking was implemented)
        result_file = output_dir / "results.json"
        if result_file.exists():
            try:
                with open(result_file) as f:
                    existing_result = json.load(f)
                if existing_result.get("success", False):
                    val = existing_result.get("val_quality")
                    log.info(f"⏭ SKIP {run_summary} | completed val={val:.4f}")
                    return val if val is not None else float("-inf")
                elif existing_result.get("singular", False):
                    val = existing_result.get("val_quality")
                    val_str = f"val={val:.4f}" if val is not None else ""
                    log.info(f"⏭ SKIP {run_summary} | singular {val_str}")
                    return val if val is not None else float("-inf")
                elif existing_result.get("oom_error", False):
                    log.info(f"♻ RETRY {run_summary} | OOM")
                    # Don't skip OOM errors - they may succeed after memory fixes
                else:
                    log.info(f"♻ RETRY {run_summary} | failed (non-singular)")
            except (json.JSONDecodeError, KeyError) as e:
                log.warning(f"Could not read previous result: {e}, re-running")

    log.info(f"▶ RUN {run_summary}")

    if is_image_model(cfg.model.name):
        if cfg.trainer.type == "cnn":
            data, dataset_info = load_image_dataset_cnn(
                cfg.dataset.name,
                n_train=cfg.dataset.get("n_train"),
                n_val=cfg.dataset.get("n_val"),
                n_test=cfg.dataset.get("n_test"),
                data_dir=cfg.get("data_dir"),
            )
            data = {k: v.to(DEVICE) for k, v in data.items()}
            model = create_image_model(cfg, dataset_info)
            result = run_cnn(cfg, model, data, output_dir)
        else:
            model_type = "cmpo3" if cfg.model.name == "CMPO3" else "cmpo2"
            data, dataset_info = load_image_dataset(
                cfg.dataset.name,
                n_patches=cfg.model.get("n_patches", 4),
                n_train=cfg.dataset.get("n_train"),
                n_val=cfg.dataset.get("n_val"),
                n_test=cfg.dataset.get("n_test"),
                model_type=model_type,
                data_dir=cfg.get("data_dir"),
                bias=True,
            )
            data = move_data_to_device(data)
            model = create_image_model(cfg, dataset_info)

            if cfg.trainer.type == "ntn":
                result = run_ntn_image(cfg, model, data, output_dir)
            elif cfg.trainer.type == "gtn":
                result = run_gtn_image(cfg, model, data, output_dir)
            else:
                raise ValueError(f"Unknown trainer type: {cfg.trainer.type}")
    else:
        data, dataset_info = load_dataset(
            cfg.dataset.name,
            csv_path=cfg.dataset.get("csv_path"),
            task=cfg.dataset.get("task"),
        )
        data = move_data_to_device(data)

        raw_feature_count = data["X_train"].shape[1]
        input_dim = raw_feature_count + 1
        output_dim = data["y_train"].shape[1] if data["y_train"].ndim > 1 else 1

        model = create_model(
            cfg, input_dim, output_dim, raw_feature_count=raw_feature_count
        )

        if cfg.trainer.type == "ntn":
            if cfg.model.name in GTN_ONLY_MODELS:
                raise ValueError(
                    f"{cfg.model.name} is GTN-only and cannot be used with NTN trainer"
                )
            result = run_ntn(cfg, model, data, output_dir)
        elif cfg.trainer.type == "gtn":
            result = run_gtn(cfg, model, data, output_dir)
        elif cfg.trainer.type == "dmrg":
            if not cfg.model.name.startswith("TNML"):
                raise ValueError(
                    f"DMRG trainer only supports TNML models, got {cfg.model.name}"
                )
            result = run_dmrg(cfg, model, data, output_dir)
        else:
            raise ValueError(f"Unknown trainer type: {cfg.trainer.type}")

    if cfg.get("save_model", False) and result.get("success", False):
        try:
            if cfg.trainer.type == "ntn":
                ntn = result["_ntn"]
                is_typei = result["_is_typei"]
                if is_typei:
                    tns = [ntn_i.tn for ntn_i in ntn.ntns]
                    quimb.save_to_disk(tns, output_dir / "model.joblib")
                else:
                    quimb.save_to_disk(ntn.tn, output_dir / "model.joblib")
            else:
                gtn_model = result["_gtn_model"]
                torch.save(gtn_model.state_dict(), output_dir / "model.pt")
            log.info(f"Saved model to {output_dir}")
        except Exception as e:
            log.warning(f"Failed to save model: {e}")

    private_keys = [k for k in result.keys() if k.startswith("_")]
    for k in private_keys:
        del result[k]

    result_file = output_dir / "results.json"
    result["config"] = OmegaConf.to_container(cfg, resolve=True)
    result["dataset_info"] = dataset_info

    with open(result_file, "w") as f:
        json.dump(result, f, indent=2, default=str)

    # Append to tracking CSV (only on main, disabled on clusters)
    if cfg.get("update_tracking", False):
        try:
            # Get relative output path from current working directory
            cwd = Path.cwd()
            try:
                relative_output = output_dir.relative_to(cwd)
            except ValueError:
                # output_dir is not relative to cwd, use absolute
                relative_output = output_dir

            append_run_result(
                result=result,
                cfg=cfg,
                output_path=relative_output,
                path=cwd / DEFAULT_TRACKING_FILE,
            )
        except Exception as e:
            log.warning(f"Failed to update tracking file: {e}")

    val_q = result.get("val_quality")
    val_str = f"val={val_q:.4f}" if val_q is not None else "val=None"
    if result["success"]:
        log.info(f"✓ DONE {run_summary} | {val_str}")
    elif result.get("oom_error", False):
        log.warning(f"✗ OOM {run_summary} | {val_str}")
    else:
        log.warning(f"✗ SINGULAR {run_summary} | {val_str}")

    model = None
    data = None
    result = None
    dataset_info = None


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    return _main_impl(cfg)


if __name__ == "__main__":
    main()
