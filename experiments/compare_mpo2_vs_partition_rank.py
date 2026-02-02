# type: ignore
"""
Compare MPO2 vs PartitionRank3 models on UCI dataset.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

from experiments.dataset_loader import load_dataset

from model.base.NTN import NTN
from model.base.NTN_Ensemble import NTN_Ensemble
from model.losses import MSELoss
from model.utils import REGRESSION_METRICS, compute_quality, create_inputs
from model.standard import MPO2
from model.partition_rank import PartitionRank3

torch.set_default_dtype(torch.float64)

JITTER_START = 5.0
JITTER_DECAY = 0.25
JITTER_MIN = 0.001
N_EPOCHS = 15
BATCH_SIZE = 32
PATIENCE = 4
MIN_DELTA = 1e-6


def run_mpo2(data, input_dim, bond_dim, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = MPO2(L=3, bond_dim=bond_dim, phys_dim=input_dim, output_dim=1)
    n_params = sum(t.size for t in model.tn.tensors)

    loader_train = create_inputs(
        X=data["X_train"],
        y=data["y_train"],
        input_labels=model.input_labels,
        output_labels=model.output_dims,
        batch_size=BATCH_SIZE,
        append_bias=False,
    )
    loader_val = create_inputs(
        X=data["X_val"],
        y=data["y_val"],
        input_labels=model.input_labels,
        output_labels=model.output_dims,
        batch_size=BATCH_SIZE,
        append_bias=False,
    )

    ntn = NTN(
        tn=model.tn,
        output_dims=model.output_dims,
        input_dims=model.input_dims,
        loss=MSELoss(),
        data_stream=loader_train,
    )

    jitter_schedule = [max(JITTER_START * (JITTER_DECAY**e), JITTER_MIN) for e in range(N_EPOCHS)]

    scores_train, scores_val = ntn.fit(
        n_epochs=N_EPOCHS,
        regularize=True,
        jitter=jitter_schedule,
        eval_metrics=REGRESSION_METRICS,
        val_data=loader_val,
        verbose=False,
        patience=PATIENCE,
        min_delta=MIN_DELTA,
        train_selection=True,
    )

    return {
        "model": "MPO2",
        "bond_dim": bond_dim,
        "partition_rank": None,
        "rank_dims": None,
        "n_params": n_params,
        "train_quality": compute_quality(scores_train),
        "val_quality": compute_quality(scores_val),
    }


def run_partition_rank(data, input_dim, partition_rank, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = PartitionRank3(phys_dim=input_dim, output_dim=1, partition_rank=partition_rank)
    n_params = model.count_parameters()

    ntn = NTN_Ensemble(
        tns=model.tns,
        input_dims_list=model.input_dims_list,
        input_labels_list=model.input_labels_list,
        output_dims=model.output_dims,
        loss=MSELoss(),
        X_train=data["X_train"],
        y_train=data["y_train"],
        X_val=data["X_val"],
        y_val=data["y_val"],
        batch_size=BATCH_SIZE,
    )

    jitter_schedule = [max(JITTER_START * (JITTER_DECAY**e), JITTER_MIN) for e in range(N_EPOCHS)]

    scores_train, scores_val = ntn.fit(
        n_epochs=N_EPOCHS,
        regularize=True,
        jitter=jitter_schedule,
        eval_metrics=REGRESSION_METRICS,
        verbose=False,
        patience=PATIENCE,
        min_delta=MIN_DELTA,
        train_selection=True,
    )

    return {
        "model": "PartitionRank3",
        "bond_dim": None,
        "partition_rank": partition_rank,
        "rank_dims": model.rank_dims,
        "n_params": n_params,
        "train_quality": compute_quality(scores_train),
        "val_quality": compute_quality(scores_val),
    }


def main():
    dataset_name = "abalone"
    max_rank = 10
    seed = 42

    print("=" * 70)
    print(f"MPO2 vs PartitionRank3 Comparison on '{dataset_name}'")
    print("=" * 70)

    data, info = load_dataset(dataset_name)
    input_dim = data["X_train"].shape[1]

    print(f"Dataset: {info['name']}")
    print(f"Train: {info['n_train']}, Val: {info['n_val']}, Test: {info['n_test']}")
    print(f"Features: {input_dim}")
    print(f"Params: jitter_start={JITTER_START}, jitter_decay={JITTER_DECAY}, epochs={N_EPOCHS}")
    print()

    results = []

    print("MPO2 (bond_dim 1-10):")
    print("-" * 60)
    for bond_dim in range(1, max_rank + 1):
        try:
            r = run_mpo2(data, input_dim, bond_dim, seed)
            results.append(r)
            print(
                f"  D={bond_dim:2d} | params={r['n_params']:6d} | train_R2={r['train_quality']:.4f} | val_R2={r['val_quality']:.4f}"
            )
        except Exception as e:
            print(f"  D={bond_dim:2d} | ERROR: {e}")

    print()
    print("PartitionRank3 (partition_rank 3-12):")
    print("-" * 60)
    for pr in range(3, max_rank + 3):
        try:
            r = run_partition_rank(data, input_dim, pr, seed)
            results.append(r)
            rd = r["rank_dims"]
            print(
                f"  PR={pr:2d} ({rd[0]},{rd[1]},{rd[2]}) | params={r['n_params']:6d} | train_R2={r['train_quality']:.4f} | val_R2={r['val_quality']:.4f}"
            )
        except Exception as e:
            print(f"  PR={pr:2d} | ERROR: {e}")

    df = pd.DataFrame(results)

    print()
    print("=" * 70)
    print("SUMMARY TABLE")
    print("=" * 70)
    print(df.to_string(index=False))

    return df


if __name__ == "__main__":
    main()
