# type: ignore
"""
Unified experiment runner with Hydra configuration.

Usage:
    python run.py                                    # defaults (MPO2, iris, NTN)
    python run.py model=lmpo2 dataset=abalone        # override model/dataset
    python run.py trainer=gtn trainer.lr=0.01        # GTN with custom LR
    python run.py --multirun model.bond_dim=4,6,8    # sweep bond dimensions
    python run.py --multirun seed=0,1,2,3,4          # multi-seed
"""

import json
import logging
from pathlib import Path

import hydra
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from omegaconf import DictConfig, OmegaConf

from utils.dataset_loader import load_dataset
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
from model.utils import (
    CLASSIFICATION_METRICS,
    REGRESSION_METRICS,
    compute_quality,
    create_inputs,
    create_inputs_tnml,
    encode_polynomial,
    encode_fourier,
)

torch.set_default_dtype(torch.float64)
log = logging.getLogger(__name__)


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
        torch.cuda.empty_cache()


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
            batch_size=cfg.dataset.batch_size,
            not_trainable_tags=getattr(model, "not_trainable_tags", None),
        )
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

    # Metrics log
    metrics_log = []

    def callback_epoch(epoch, scores_train, scores_val, info):
        metrics = {
            "epoch": epoch,
            "train_loss": float(scores_train["loss"]),
            "train_quality": float(compute_quality(scores_train)),
            "val_loss": float(scores_val["loss"]),
            "val_quality": float(compute_quality(scores_val)),
            "ridge": float(info["jitter"]),
        }
        metrics_log.append(metrics)

    # Train
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
        # Clear CUDA cache to allow graceful exit
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        scores_train = None
        scores_val = None

    # Extract best metrics from metrics_log
    # For successful runs: NTN.fit returns best scores, but we also track best_epoch
    # For singular runs: extract best values achieved before failure
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

    # For successful runs, use scores from ntn.fit (which are the best)
    if success and scores_train is not None:
        best_train_loss = float(scores_train["loss"])
        best_train_quality = float(compute_quality(scores_train))
        best_val_loss = float(scores_val["loss"])
        best_val_quality = float(compute_quality(scores_val))

    # Capture GPU memory info after training
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
        "gpu_memory": {
            "before": gpu_info_before,
            "after": gpu_info_after,
        },
    }

    return result


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

    # Training loop
    metrics_log = []
    best_val_quality = float("-inf")
    best_train_quality = float("-inf")
    best_epoch = -1
    patience_counter = 0
    train_selection = cfg.trainer.get("train_selection", False)

    for epoch in range(n_epochs):
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

        metrics_log.append(
            {
                "epoch": epoch,
                "train_loss": float(train_loss),
                "train_quality": float(train_quality),
                "val_loss": float(val_loss),
                "val_quality": float(val_quality),
            }
        )

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
    else:
        # Fallback to final evaluation if no best epoch found
        best_train_loss, best_train_quality = evaluate(train_loader)
        best_val_loss, best_val_quality = evaluate(val_loader)

    return {
        "success": True,
        "singular": False,
        "train_loss": float(best_train_loss),
        "train_quality": float(best_train_quality),
        "val_loss": float(best_val_loss),
        "val_quality": float(best_val_quality),
        "best_epoch": best_epoch,
        "metrics_log": metrics_log,
    }


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    """Main entry point."""
    log.info(f"Configuration:\n{OmegaConf.to_yaml(cfg)}")

    # Set seeds
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    # Output directory (managed by Hydra)
    output_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    log.info(f"Output directory: {output_dir}")

    # Check for completed run (skip if already successful)
    skip_completed = cfg.get("skip_completed", True)

    if skip_completed:
        # First check tracking CSV (fast, no filesystem access to outputs)
        run_id = generate_run_id(cfg)
        tracking_df = get_tracking_df()
        skip, reason, cached_val = should_skip_run(tracking_df, run_id)

        if skip:
            log.info(f"⏭ Skipping [{run_id}]: {reason}")
            return cached_val if cached_val is not None else float("-inf")
        elif cached_val is not None:
            # Not skipping but we have previous info (e.g., OOM retry)
            log.info(f"♻ Re-running [{run_id}]: {reason}")

        # Fallback: check results.json for backwards compatibility
        # (handles runs completed before tracking was implemented)
        result_file = output_dir / "results.json"
        if result_file.exists():
            try:
                with open(result_file) as f:
                    existing_result = json.load(f)
                if existing_result.get("success", False):
                    log.info(
                        f"⏭ Skipping: already completed successfully (from results.json)"
                    )
                    log.info(
                        f"  Previous val_quality: {existing_result.get('val_quality')}"
                    )
                    return existing_result.get("val_quality", float("-inf"))
                elif existing_result.get("singular", False):
                    best_val = existing_result.get("val_quality")
                    log.info(
                        f"⏭ Skipping: previous run failed with singular matrix (permanent failure)"
                    )
                    log.info(f"  Best val_quality before failure: {best_val}")
                    return best_val if best_val is not None else float("-inf")
                elif existing_result.get("oom_error", False):
                    best_val = existing_result.get("val_quality")
                    log.info(
                        f"♻ Re-running: previous run failed with OOM error (may succeed with memory optimizations)"
                    )
                    log.info(f"  Best val_quality before failure: {best_val}")
                    # Don't skip OOM errors - they may succeed after memory fixes
                else:
                    log.info(f"♻ Re-running: previous run failed (non-singular)")
            except (json.JSONDecodeError, KeyError) as e:
                log.warning(f"Could not read previous result: {e}, re-running")

    # Load dataset
    log.info(f"Loading dataset: {cfg.dataset.name}")
    data, dataset_info = load_dataset(cfg.dataset.name)
    data = move_data_to_device(data)

    raw_feature_count = data["X_train"].shape[1]
    input_dim = raw_feature_count + 1
    output_dim = data["y_train"].shape[1] if data["y_train"].ndim > 1 else 1

    log.info(
        f"  Samples: train={dataset_info['n_train']}, val={dataset_info['n_val']}, test={dataset_info['n_test']}"
    )
    log.info(f"  Features: {input_dim} (including bias), Task: {dataset_info['task']}")
    log.info(f"  Device: {DEVICE}")

    # Create model
    log.info(f"Creating model: {cfg.model.name}")
    model = create_model(
        cfg, input_dim, output_dim, raw_feature_count=raw_feature_count
    )

    # Run training
    if cfg.trainer.type == "ntn":
        if cfg.model.name in GTN_ONLY_MODELS:
            raise ValueError(
                f"{cfg.model.name} is GTN-only and cannot be used with NTN trainer"
            )
        result = run_ntn(cfg, model, data, output_dir)
    elif cfg.trainer.type == "gtn":
        result = run_gtn(cfg, model, data, output_dir)
    else:
        raise ValueError(f"Unknown trainer type: {cfg.trainer.type}")

    # Save results
    result_file = output_dir / "results.json"
    result["config"] = OmegaConf.to_container(cfg, resolve=True)
    result["dataset_info"] = dataset_info

    with open(result_file, "w") as f:
        json.dump(result, f, indent=2, default=str)

    log.info(f"Results saved to: {result_file}")

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

    # Log summary
    if result["success"]:
        quality_name = "R²" if cfg.dataset.task == "regression" else "Accuracy"
        log.info(f"✓ Train {quality_name}: {result['train_quality']:.4f}")
        log.info(f"✓ Val {quality_name}: {result['val_quality']:.4f}")
    elif result.get("oom_error", False):
        log.warning("✗ Training failed (CUDA out of memory)")
        if result.get("val_quality") is not None:
            log.info(f"  Best val_quality before failure: {result['val_quality']:.4f}")
    else:
        log.warning("✗ Training failed (singular matrix)")

    # Return val_quality for Hydra optimization
    return result.get("val_quality", float("-inf"))


if __name__ == "__main__":
    main()
