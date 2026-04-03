"""Run tracking utilities for distributed cluster experiments.

This module provides CSV-based tracking of experiment runs to enable:
- Skipping already-completed runs across multiple clusters
- Tracking run outcomes (success, singular, OOM) without checking output files
- Easy merging of results between cluster branches

Schema:
    run_id: Deterministic identifier from config params
    trainer_type: ntn or gtn
    dataset: Dataset name
    model: Model name
    L: Number of sites
    bond_dim: Bond dimension
    ridge: Ridge regularization value
    init_strength: Initialization strength
    seed: Random seed
    success: Whether run completed successfully
    singular: Whether run failed due to singular matrix
    oom_error: Whether run failed due to OOM
    val_quality: Best validation quality achieved
    train_quality: Best training quality achieved
    best_epoch: Epoch of best validation
    output_path: Relative path to results directory
    timestamp: ISO timestamp of run completion
"""

from __future__ import annotations

import csv
import fcntl
import logging
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from omegaconf import DictConfig

log = logging.getLogger(__name__)

# CSV columns in order
TRACKING_COLUMNS = [
    "run_id",
    "trainer_type",
    "dataset",
    "model",
    "L",
    "bond_dim",
    "ridge",
    "init_strength",
    "seed",
    "success",
    "singular",
    "oom_error",
    "val_quality",
    "train_quality",
    "best_epoch",
    "output_path",
    "timestamp",
]

DEFAULT_TRACKING_FILE = "runs_tracking.csv"


def generate_run_id(cfg: DictConfig) -> str:
    """Generate a deterministic run ID from config parameters.

    Format: {trainer}_{dataset}_{model}_L{L}_bd{bond_dim}_rg{ridge}_init{init}_s{seed}

    Args:
        cfg: Hydra configuration object

    Returns:
        Unique run identifier string
    """
    return (
        f"{cfg.trainer.type}_{cfg.dataset.name}_{cfg.model.name}"
        f"_L{cfg.model.L}_bd{cfg.model.bond_dim}"
        f"_rg{cfg.trainer.ridge}_init{cfg.model.init_strength}"
        f"_s{cfg.seed}"
    )


def load_tracking_file(path: str | Path = DEFAULT_TRACKING_FILE) -> pd.DataFrame:
    """Load the tracking CSV file into a DataFrame.

    If the file doesn't exist, returns an empty DataFrame with correct columns.

    Args:
        path: Path to the tracking CSV file

    Returns:
        DataFrame with tracking data, indexed by run_id
    """
    path = Path(path)

    if not path.exists():
        log.info(f"Tracking file not found at {path}, starting fresh")
        df = pd.DataFrame(columns=TRACKING_COLUMNS)
        df.set_index("run_id", inplace=True)
        return df

    try:
        df = pd.read_csv(path, dtype={"seed": int, "L": int, "bond_dim": int})
        df.set_index("run_id", inplace=True)
        log.info(f"Loaded {len(df)} tracked runs from {path}")
        return df
    except Exception as e:
        log.warning(f"Error loading tracking file: {e}, starting fresh")
        df = pd.DataFrame(columns=TRACKING_COLUMNS)
        df.set_index("run_id", inplace=True)
        return df


def get_run_status(df: pd.DataFrame, run_id: str) -> dict | None:
    """Get the status of a specific run from the tracking DataFrame.

    Args:
        df: Tracking DataFrame (indexed by run_id)
        run_id: The run identifier to look up

    Returns:
        Dict with run status fields, or None if not found
    """
    if run_id not in df.index:
        return None

    row = df.loc[run_id]
    return row.to_dict()


def should_skip_run(df: pd.DataFrame, run_id: str) -> tuple[bool, str, float | None]:
    """Determine if a run should be skipped based on tracking data.

    Skip logic:
    - success=True: Skip (already completed successfully)
    - singular=True: Skip (permanent failure, won't succeed on retry)
    - oom_error=True: Don't skip (may succeed with memory optimizations)
    - Not in tracking: Don't skip (new run)

    Args:
        df: Tracking DataFrame (indexed by run_id)
        run_id: The run identifier to check

    Returns:
        Tuple of (should_skip, reason, val_quality)
        - should_skip: True if run should be skipped
        - reason: Human-readable reason for skipping/running
        - val_quality: Previous val_quality if available, else None
    """
    status = get_run_status(df, run_id)

    if status is None:
        return False, "not in tracking file", None

    val_quality = status.get("val_quality")
    if pd.isna(val_quality):
        val_quality = None

    if status.get("success"):
        return True, f"already completed successfully (val_quality={val_quality})", val_quality

    if status.get("singular"):
        return True, f"previous run failed with singular matrix (permanent)", val_quality

    if status.get("oom_error"):
        return False, f"previous OOM error, retrying (best val_quality={val_quality})", val_quality

    # Unknown failure state - retry
    return False, "previous run failed (non-singular), retrying", val_quality


def append_run_result(
    result: dict,
    cfg: DictConfig,
    output_path: str | Path,
    path: str | Path = DEFAULT_TRACKING_FILE,
) -> None:
    """Append a run result to the tracking CSV file.

    Uses file locking to prevent corruption from concurrent writes
    (though this should be rare in batch job scenarios).

    Args:
        result: Dict with run results (success, singular, oom_error, val_quality, etc.)
        cfg: Hydra configuration object
        output_path: Path to the output directory for this run
        path: Path to the tracking CSV file
    """
    path = Path(path)
    run_id = generate_run_id(cfg)

    row = {
        "run_id": run_id,
        "trainer_type": cfg.trainer.type,
        "dataset": cfg.dataset.name,
        "model": cfg.model.name,
        "L": cfg.model.L,
        "bond_dim": cfg.model.bond_dim,
        "ridge": cfg.trainer.ridge,
        "init_strength": cfg.model.init_strength,
        "seed": cfg.seed,
        "success": result.get("success", False),
        "singular": result.get("singular", False),
        "oom_error": result.get("oom_error", False),
        "val_quality": result.get("val_quality"),
        "train_quality": result.get("train_quality"),
        "best_epoch": result.get("best_epoch"),
        "output_path": str(output_path),
        "timestamp": datetime.now().isoformat(),
    }

    file_exists = path.exists()

    try:
        with open(path, "a", newline="") as f:
            # Acquire exclusive lock for writing
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            try:
                writer = csv.DictWriter(f, fieldnames=TRACKING_COLUMNS)
                if not file_exists:
                    writer.writeheader()
                writer.writerow(row)
                log.info(f"Appended run {run_id} to tracking file")
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)
    except Exception as e:
        log.error(f"Failed to append to tracking file: {e}")
        raise


def create_tracking_file(path: str | Path = DEFAULT_TRACKING_FILE) -> None:
    """Create an empty tracking file with headers.

    Args:
        path: Path to create the tracking file at
    """
    path = Path(path)

    if path.exists():
        log.info(f"Tracking file already exists at {path}")
        return

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=TRACKING_COLUMNS)
        writer.writeheader()

    log.info(f"Created tracking file at {path}")


def get_tracking_summary(df: pd.DataFrame) -> dict:
    """Get a summary of the tracking data.

    Args:
        df: Tracking DataFrame

    Returns:
        Dict with summary statistics
    """
    return {
        "total_runs": len(df),
        "successful": df["success"].sum() if "success" in df.columns else 0,
        "singular": df["singular"].sum() if "singular" in df.columns else 0,
        "oom_errors": df["oom_error"].sum() if "oom_error" in df.columns else 0,
        "datasets": df["dataset"].nunique() if "dataset" in df.columns else 0,
        "models": df["model"].nunique() if "model" in df.columns else 0,
    }
