# type: ignore
import json
import logging
import os
import sys
from pathlib import Path

import hydra
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from omegaconf import DictConfig, OmegaConf
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments_imgs.image_dataset_loader import load_image_dataset
from experiments_imgs.models import CMPO2, CMPO3, CMPO2_GTN, CMPO3_GTN
from utils.device_utils import DEVICE, move_data_to_device, move_tn_to_device


torch.set_default_dtype(torch.float64)
log = logging.getLogger(__name__)


def create_cmpo2(cfg: DictConfig, dataset_info: dict) -> CMPO2:
    return CMPO2(
        L=cfg.model.L,
        pixel_dim=dataset_info["pixels_per_patch"],
        patch_dim=dataset_info["n_patches"],
        pixel_bond_dim=cfg.model.rank_pixel,
        patch_bond_dim=cfg.model.rank_patch,
        output_dim=dataset_info["n_classes"],
        init_strength=cfg.model.init_strength,
    )


def create_cmpo3(cfg: DictConfig, dataset_info: dict) -> CMPO3:
    return CMPO3(
        L=cfg.model.L,
        channel_dim=dataset_info["n_channels"],
        pixel_dim=dataset_info["pixels_per_patch"],
        patch_dim=dataset_info["n_patches"],
        channel_bond_dim=cfg.model.rank_channel,
        pixel_bond_dim=cfg.model.rank_pixel,
        patch_bond_dim=cfg.model.rank_patch,
        output_dim=dataset_info["n_classes"],
        init_strength=cfg.model.init_strength,
    )


def create_optimizer(
    name: str, parameters, lr: float, weight_decay: float
) -> optim.Optimizer:
    name = name.lower()
    if name == "adam":
        return optim.Adam(parameters, lr=lr, weight_decay=weight_decay)
    elif name == "adamw":
        return optim.AdamW(parameters, lr=lr, weight_decay=weight_decay)
    elif name == "sgd":
        return optim.SGD(parameters, lr=lr, weight_decay=weight_decay, momentum=0.9)
    else:
        raise ValueError(f"Unknown optimizer: {name}")


def evaluate(model: nn.Module, loader, criterion) -> tuple[float, float]:
    model.eval()
    total_loss = 0
    correct, total = 0, 0

    with torch.no_grad():
        for batch_data, batch_target in loader:
            batch_data, batch_target = batch_data, batch_target
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


def run_gtn(cfg: DictConfig, model: nn.Module, data: dict, output_dir: Path) -> dict:
    criterion = nn.CrossEntropyLoss()
    optimizer = create_optimizer(
        cfg.trainer.optimizer,
        model.parameters(),
        cfg.trainer.lr,
        cfg.trainer.weight_decay,
    )

    batch_size = cfg.dataset.batch_size
    n_epochs = cfg.trainer.n_epochs
    patience = cfg.trainer.patience
    min_delta = cfg.trainer.min_delta

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

    metrics_log = []
    best_val_accuracy = 0.0
    best_epoch = -1
    patience_counter = 0
    stopped_early = False

    for epoch in range(n_epochs):
        model.train()
        train_loss = 0.0

        for batch_data, batch_target in train_loader:
            batch_data, batch_target = batch_data, batch_target
            optimizer.zero_grad()
            output = model(batch_data)
            loss = criterion(output, batch_target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * batch_data.size(0)

        train_loss /= len(train_loader.dataset)
        _, train_accuracy = evaluate(model, train_loader, criterion)
        val_loss, val_accuracy = evaluate(model, val_loader, criterion)

        metrics_log.append(
            {
                "epoch": epoch,
                "train_loss": float(train_loss),
                "train_accuracy": float(train_accuracy),
                "val_loss": float(val_loss),
                "val_accuracy": float(val_accuracy),
            }
        )

        if val_accuracy > best_val_accuracy + min_delta:
            best_val_accuracy = val_accuracy
            best_epoch = epoch
            patience_counter = 0
        else:
            patience_counter += 1

        if patience is not None and patience_counter >= patience:
            log.info(f"Early stopping at epoch {epoch + 1}")
            stopped_early = True
            break

        if epoch % 10 == 0 or epoch == n_epochs - 1:
            log.info(
                f"Epoch {epoch + 1:3d} | Train Loss: {train_loss:.4f} | Train Acc: {train_accuracy:.4f} | Val Acc: {val_accuracy:.4f}"
            )

    test_loss, test_accuracy = evaluate(model, test_loader, criterion)

    return {
        "success": True,
        "singular": False,
        "train_loss": float(train_loss),
        "train_accuracy": float(train_accuracy),
        "val_loss": float(val_loss),
        "val_accuracy": float(best_val_accuracy),
        "test_loss": float(test_loss),
        "test_accuracy": float(test_accuracy),
        "best_epoch": best_epoch,
        "stopped_early": stopped_early,
        "metrics_log": metrics_log,
    }


def create_inputs_cmpo(X, y, input_labels, output_labels, batch_size, n_patches):
    from model.builder import Inputs

    return Inputs(
        inputs=[X],
        outputs=[y],
        outputs_labels=output_labels,
        input_labels=input_labels,
        batch_dim="s",
        batch_size=batch_size,
    )


def run_ntn(cfg: DictConfig, cmpo_model, data: dict, output_dir: Path) -> dict:
    from model.base.NTN import NTN
    from model.losses import CrossEntropyLoss
    from model.utils import CLASSIFICATION_METRICS, compute_quality

    move_tn_to_device(cmpo_model.tn)

    n_epochs = cfg.trainer.n_epochs
    batch_size = cfg.dataset.batch_size
    n_patches = cfg.model.n_patches

    ridge_schedule = [
        max(cfg.trainer.ridge * (cfg.trainer.ridge_decay**epoch), cfg.trainer.ridge_min)
        for epoch in range(n_epochs)
    ]

    loader_train = create_inputs_cmpo(
        data["X_train"],
        data["y_train"],
        cmpo_model.input_labels,
        cmpo_model.output_dims,
        batch_size,
        n_patches,
    )

    loader_val = create_inputs_cmpo(
        data["X_val"],
        data["y_val"],
        cmpo_model.input_labels,
        cmpo_model.output_dims,
        batch_size,
        n_patches,
    )

    loss_fn = CrossEntropyLoss()

    ntn = NTN(
        tn=cmpo_model.tn,
        output_dims=cmpo_model.output_dims,
        input_dims=cmpo_model.input_dims,
        loss=loss_fn,
        data_stream=loader_train,
    )

    metrics_log = []

    def callback_epoch(epoch, scores_train, scores_val, info):
        metrics = {
            "epoch": epoch,
            "train_loss": float(scores_train["loss"]),
            "train_accuracy": float(compute_quality(scores_train)),
            "val_loss": float(scores_val["loss"]),
            "val_accuracy": float(compute_quality(scores_val)),
            "ridge": float(info["jitter"]),
        }
        metrics_log.append(metrics)

    try:
        trainable_nodes = ntn._get_trainable_nodes()
        groups = defaultdict(list)
        for node in trainable_nodes:
            groups[int(node.split('_')[0])].append(node)

        indices = sorted(groups.keys())
        sequence = indices + indices[-2:0:-1]

        ordered_list = [node for i in sequence for node in groups[i]]
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
    except Exception as e:
        log.error(f"NTN training failed: {e}")
        success = False
        singular = True
        scores_train = None
        scores_val = None

    best_epoch = -1
    best_val_accuracy = 0.0
    if metrics_log:
        for m in metrics_log:
            if m["val_accuracy"] > best_val_accuracy:
                best_val_accuracy = m["val_accuracy"]
                best_epoch = m["epoch"]

    if success and scores_train is not None:
        train_accuracy = float(compute_quality(scores_train))
        val_accuracy = float(compute_quality(scores_val))
    else:
        train_accuracy = metrics_log[-1]["train_accuracy"] if metrics_log else 0.0
        val_accuracy = best_val_accuracy

    loader_test = create_inputs_cmpo(
        data["X_test"],
        data["y_test"],
        cmpo_model.input_labels,
        cmpo_model.output_dims,
        batch_size,
        n_patches,
    )
    if success:
        scores_test = ntn.evaluate(CLASSIFICATION_METRICS, data=loader_test)
        test_accuracy = float(compute_quality(scores_test))
    else:
        test_accuracy = 0.0

    return {
        "success": success,
        "singular": singular,
        "train_accuracy": train_accuracy,
        "val_accuracy": val_accuracy,
        "test_accuracy": test_accuracy,
        "best_epoch": best_epoch,
        "metrics_log": metrics_log,
    }


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    log.info(f"Configuration:\n{OmegaConf.to_yaml(cfg)}")

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    output_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    log.info(f"Output directory: {output_dir}")

    if cfg.skip_completed:
        result_file = output_dir / "results.json"
        if result_file.exists():
            try:
                with open(result_file) as f:
                    existing = json.load(f)
                if existing.get("success", False):
                    log.info(
                        f"Skipping: already completed. Test acc: {existing.get('test_accuracy')}"
                    )
                    return existing.get("test_accuracy", float("-inf"))
                elif existing.get("singular", False):
                    log.info(f"Skipping: singular matrix failure")
                    return existing.get("test_accuracy", float("-inf"))
            except (json.JSONDecodeError, KeyError):
                pass

    model_type = "cmpo3" if cfg.model.name == "CMPO3" else "cmpo2"
    n_patches = cfg.model.n_patches
    n_train = cfg.dataset.get("n_train")
    n_val = cfg.dataset.get("n_val")
    n_test = cfg.dataset.get("n_test")
    data_dir = cfg.get("data_dir")

    log.info(f"Loading dataset: {cfg.dataset.name} (n_patches={n_patches})")
    data, dataset_info = load_image_dataset(
        cfg.dataset.name,
        n_patches=n_patches,
        n_train=n_train,
        n_val=n_val,
        n_test=n_test,
        model_type=model_type,
        data_dir=data_dir,
        bias=True,
    )

    data = move_data_to_device(data)

    log.info(
        f"  Train: {dataset_info['n_train']}, Val: {dataset_info['n_val']}, Test: {dataset_info['n_test']}"
    )
    log.info(
        f"  Patches: {n_patches}, Pixels/patch: {dataset_info['pixels_per_patch']}"
    )
    log.info(f"  Device: {DEVICE}")

    log.info(f"Creating model: {cfg.model.name}")
    if cfg.model.name == "CMPO2":
        cmpo_model = create_cmpo2(cfg, dataset_info)
    elif cfg.model.name == "CMPO3":
        cmpo_model = create_cmpo3(cfg, dataset_info)
    else:
        raise ValueError(f"Unknown model: {cfg.model.name}")

    if cfg.trainer.type == "gtn":
        model = (
            CMPO2_GTN(cmpo_model)
            if cfg.model.name == "CMPO2"
            else CMPO3_GTN(cmpo_model)
        )
        model = model.to(DEVICE)
        n_params = sum(p.numel() for p in model.parameters())
        log.info(f"  Parameters: {n_params:,}")
        result = run_gtn(cfg, model, data, output_dir)
        result["n_parameters"] = n_params

    elif cfg.trainer.type == "ntn":
        n_params = sum(t.data.numel() for t in cmpo_model.tn.tensors)
        log.info(f"  Parameters: {n_params:,}")
        result = run_ntn(cfg, cmpo_model, data, output_dir)
        result["n_parameters"] = n_params

    else:
        raise ValueError(f"Unknown trainer type: {cfg.trainer.type}")

    result["config"] = OmegaConf.to_container(cfg, resolve=True)
    result["dataset_info"] = dataset_info

    result_file = output_dir / "results.json"
    with open(result_file, "w") as f:
        json.dump(result, f, indent=2, default=str)

    log.info(f"Results saved to: {result_file}")
    log.info(f"Test Accuracy: {result['test_accuracy']:.4f}")

    return result["test_accuracy"]

if __name__ == "__main__":
    main()
