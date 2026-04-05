#!/usr/bin/env python3
"""Compare TNML environment behavior with MPO2."""

import sys
sys.path.insert(0, '.')

import torch
from model.standard.TNML import TNML_P
from model.standard.MPO2_models import MPO2
from model.utils import create_inputs, create_inputs_tnml
from model.base.NTN import NTN
from model.losses import MSELoss


def compare_environments():
    print("=" * 60)
    print("Comparing Environment Computation: MPO2 vs TNML_P")
    print("=" * 60)
    
    n_samples = 16
    L = 3
    bond_dim = 4
    output_dim = 2
    batch_size = 8
    
    # MPO2: L nodes, phys_dim = n_features+1 (with bias)
    n_features_mpo2 = 4
    phys_dim_mpo2 = n_features_mpo2 + 1
    
    X_mpo2 = torch.randn(n_samples, n_features_mpo2)
    y = torch.randn(n_samples, output_dim)
    
    mpo2 = MPO2(
        L=L,
        bond_dim=bond_dim,
        phys_dim=phys_dim_mpo2,
        output_dim=output_dim,
        use_tn_normalization=False,
        init_strength=0.1,
    )
    
    loader_mpo2 = create_inputs(
        X=X_mpo2,
        y=y,
        input_labels=mpo2.input_labels,
        output_labels=mpo2.output_dims,
        batch_size=batch_size,
        append_bias=True,
    )
    
    ntn_mpo2 = NTN(
        tn=mpo2.tn,
        output_dims=mpo2.output_dims,
        input_dims=mpo2.input_dims,
        loss=MSELoss(),
        data_stream=loader_mpo2,
    )
    
    print("\n--- MPO2 ---")
    print(f"L={L}, phys_dim={phys_dim_mpo2}, bond_dim={bond_dim}")
    print(f"input_labels: {mpo2.input_labels}")
    print(f"output_dims: {mpo2.output_dims}")
    
    mu_mpo2, _ = loader_mpo2[0]
    print(f"\nInput tensors: {len(mu_mpo2)}")
    for i, inp in enumerate(mu_mpo2):
        print(f"  Input {i}: shape={inp.shape}, inds={inp.inds}")
    
    print(f"\nModel tensors:")
    for tensor in mpo2.tn:
        print(f"  {tensor.tags}: shape={tensor.shape}, inds={tensor.inds}")
    
    print(f"\nEnvironments:")
    for tensor in mpo2.tn:
        tag = list(tensor.tags)[0]
        env = ntn_mpo2._batch_environment(mu_mpo2, ntn_mpo2.tn, tag)
        print(f"  {tag}: tensor_inds={tensor.inds}, env_inds={env.inds}")
    
    # TNML_P: n_features nodes, phys_dim = L+1
    n_features_tnml = 4
    degree = 2
    
    X_tnml = torch.randn(n_samples, n_features_tnml)
    
    tnml = TNML_P(
        L=degree,
        bond_dim=bond_dim,
        phys_dim=n_features_tnml,
        output_dim=output_dim,
        use_tn_normalization=False,
        init_strength=0.1,
    )
    
    loader_tnml = create_inputs_tnml(
        X=X_tnml,
        y=y,
        input_labels=tnml.input_labels,
        output_labels=tnml.output_dims,
        batch_size=batch_size,
        encoding="polynomial",
        degree=degree,
    )
    
    ntn_tnml = NTN(
        tn=tnml.tn,
        output_dims=tnml.output_dims,
        input_dims=tnml.input_dims,
        loss=MSELoss(),
        data_stream=loader_tnml,
    )
    
    print("\n--- TNML_P ---")
    print(f"n_features={n_features_tnml}, degree={degree}, bond_dim={bond_dim}")
    print(f"input_labels: {tnml.input_labels}")
    print(f"output_dims: {tnml.output_dims}")
    
    mu_tnml, _ = loader_tnml[0]
    print(f"\nInput tensors: {len(mu_tnml)}")
    for i, inp in enumerate(mu_tnml):
        print(f"  Input {i}: shape={inp.shape}, inds={inp.inds}")
    
    print(f"\nModel tensors:")
    for tensor in tnml.tn:
        print(f"  {tensor.tags}: shape={tensor.shape}, inds={tensor.inds}")
    
    print(f"\nEnvironments:")
    for tensor in tnml.tn:
        tag = list(tensor.tags)[0]
        env = ntn_tnml._batch_environment(mu_tnml, ntn_tnml.tn, tag)
        print(f"  {tag}: tensor_inds={tensor.inds}, env_inds={env.inds}")
    
    print("\n" + "=" * 60)
    print("Testing single NTN update on both models")
    print("=" * 60)
    
    print("\nMPO2 update Node0:")
    try:
        ntn_mpo2.update_tn_node("Node0", regularize=True, jitter=1e-4)
        print("  [OK] Update succeeded")
    except Exception as e:
        print(f"  [FAIL] {e}")
    
    print("\nTNML_P update Node0:")
    try:
        ntn_tnml.update_tn_node("Node0", regularize=True, jitter=1e-4)
        print("  [OK] Update succeeded")
    except Exception as e:
        print(f"  [FAIL] {e}")


if __name__ == "__main__":
    compare_environments()
