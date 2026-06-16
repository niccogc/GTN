#!/usr/bin/env python
"""Debug script to understand why TNML models are slow on large datasets."""

import os
import time
import torch
import numpy as np

# Enable debug output in NTN
os.environ["NTN_DEBUG"] = "1"

torch.set_default_dtype(torch.float64)

from model.load_ucirepo import get_ucidata
from model.standard import TNML_F
from model.losses import MSELoss
from model.base.NTN import NTN
from model.utils import create_inputs

def main():
    print("=" * 60)
    print("TNML_F Debug Script - popularity dataset")
    print("=" * 60)
    
    # Load dataset (use CPU for debugging)
    device = "cpu"
    print(f"\n[1] Loading popularity dataset on {device}...")
    t0 = time.perf_counter()
    X_train, y_train, X_val, y_val, X_test, y_test = get_ucidata(332, 'regression', device=device)
    print(f"    Loaded in {time.perf_counter() - t0:.2f}s")
    print(f"    X_train: {X_train.shape}, y_train: {y_train.shape}")
    
    n_features = X_train.shape[1]
    n_samples = X_train.shape[0]
    
    # Use small subset for faster debugging
    use_subset = True
    if use_subset:
        n_subset = 500  # Use only 500 samples for debugging
        X_train = X_train[:n_subset]
        y_train = y_train[:n_subset]
        print(f"    Using subset: {n_subset} samples")
    
    # Model config from best_conf/ntn/tnml_f.yaml for popularity
    L = 4
    bond_dim = 8
    output_dim = 1
    batch_size = 128
    
    print(f"\n[2] Creating TNML_F model...")
    print(f"    L={L}, bond_dim={bond_dim}, n_features={n_features}")
    t0 = time.perf_counter()
    model = TNML_F(
        L=L,
        bond_dim=bond_dim,
        phys_dim=n_features,  # This becomes n_sites
        output_dim=output_dim,
    )
    print(f"    Created in {time.perf_counter() - t0:.2f}s")
    print(f"    Number of nodes: {len(list(model.tn.tensors))}")
    print(f"    Model encoding: {model.encoding}")
    print(f"    phys_dim (per site): {model.phys_dim}")
    
    # Print tensor shapes
    print(f"\n[3] Tensor shapes:")
    for i, t in enumerate(model.tn.tensors):
        print(f"    Node{i}: shape={t.shape}, inds={t.inds}")
        if i >= 5:
            print(f"    ... ({len(list(model.tn.tensors)) - 6} more nodes)")
            break
    
    # Create data loader
    print(f"\n[4] Creating data loader with batch_size={batch_size}...")
    loader_train = create_inputs(
        X=X_train,
        y=y_train,
        input_labels=model.input_labels,
        output_labels=model.output_dims,
        batch_size=batch_size,
        append_bias=False,
        encoding=model.encoding,
        poly_degree=getattr(model, 'poly_degree', None),
    )
    print(f"    Number of batches: {len(list(loader_train.data_mu_y))}")
    
    # Create NTN
    print(f"\n[5] Creating NTN trainer...")
    loss_fn = MSELoss()
    ntn = NTN(
        tn=model.tn,
        output_dims=model.output_dims,
        input_dims=model.input_dims,
        loss=loss_fn,
        data_stream=loader_train,
    )
    
    trainable_nodes = ntn._get_trainable_nodes()
    print(f"    Trainable nodes: {len(trainable_nodes)}")
    print(f"    Full sweep order length: {len(trainable_nodes) + len(trainable_nodes) - 2}")
    
    # Test timing of key operations
    print(f"\n[6] Timing single node derivative computation...")
    
    # Get first batch
    first_batch = next(iter(loader_train.data_mu_y))
    inputs, y_true = first_batch
    print(f"    First batch: inputs has {len(inputs)} tensors, y_true shape={y_true.shape}")
    
    # Time _batch_node_derivatives for first few nodes
    for node_idx in [0, 1, len(trainable_nodes)//2, len(trainable_nodes)-1]:
        node_tag = trainable_nodes[node_idx]
        print(f"\n    Testing {node_tag}...")
        
        t0 = time.perf_counter()
        grad, hess = ntn._batch_node_derivatives(inputs, y_true, node_tag)
        t1 = time.perf_counter()
        
        print(f"      Time: {t1-t0:.3f}s")
        print(f"      Grad shape: {grad.shape}")
        print(f"      Hess shape: {hess.shape}")
    
    # Try one full node update
    print(f"\n[7] Timing full node update (all batches)...")
    node_tag = trainable_nodes[0]
    t0 = time.perf_counter()
    
    # Recreate loader since we consumed it
    loader_train = create_inputs(
        X=X_train,
        y=y_train,
        input_labels=model.input_labels,
        output_labels=model.output_dims,
        batch_size=batch_size,
        append_bias=False,
        encoding=model.encoding,
        poly_degree=getattr(model, 'poly_degree', None),
    )
    ntn.data = loader_train
    ntn.train_data = loader_train
    
    print(f"    Computing _compute_H_b for {node_tag}...")
    t1 = time.perf_counter()
    J, H = ntn._compute_H_b(node_tag)
    t2 = time.perf_counter()
    print(f"      _compute_H_b took: {t2-t1:.3f}s")
    print(f"      J shape: {J.shape}, H shape: {H.shape}")
    
    print(f"\n[8] Estimating total epoch time...")
    n_nodes = len(trainable_nodes)
    sweep_length = n_nodes + (n_nodes - 2)  # forward + backward
    time_per_node = t2 - t1
    estimated_epoch_time = sweep_length * time_per_node
    print(f"    Nodes in sweep: {sweep_length}")
    print(f"    Time per node (estimate): {time_per_node:.3f}s")
    print(f"    Estimated epoch time: {estimated_epoch_time:.1f}s ({estimated_epoch_time/60:.1f} min)")
    
    # Scale to full dataset
    if use_subset:
        scale_factor = n_samples / n_subset
        full_epoch_estimate = estimated_epoch_time * scale_factor
        print(f"\n    With full dataset ({n_samples} samples):")
        print(f"    Estimated epoch time: {full_epoch_estimate:.1f}s ({full_epoch_estimate/60:.1f} min, {full_epoch_estimate/3600:.1f} hrs)")

    print("\n" + "=" * 60)
    print("Debug complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
