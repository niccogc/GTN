#!/usr/bin/env python3
"""Test TNML_P and TNML_F model structures."""

import sys
sys.path.insert(0, '.')

import torch
from model.standard.TNML import TNML_P, TNML_F


def test_tnml_p_structure():
    """Test TNML_P tensor network structure and bond dimensions."""
    print("=" * 60)
    print("Testing TNML_P (Polynomial)")
    print("=" * 60)
    
    n_features = 4
    L = 3
    bond_dim = 6
    output_dim = 2
    
    model = TNML_P(
        L=L,
        bond_dim=bond_dim,
        phys_dim=n_features,
        output_dim=output_dim,
        use_tn_normalization=False,
    )
    
    print(f"\nParameters: n_features={n_features}, L={L}, bond_dim={bond_dim}, output_dim={output_dim}")
    print(f"Expected: {n_features} nodes, each with phys_dim={L+1}")
    print(f"\nModel attributes:")
    print(f"  model.n_features = {model.n_features}")
    print(f"  model.phys_dim = {model.phys_dim}")
    print(f"  model.poly_degree = {model.poly_degree}")
    print(f"  model.bond_dim = {model.bond_dim}")
    print(f"  model.output_site = {model.output_site}")
    print(f"  model.input_labels = {model.input_labels}")
    print(f"  model.input_dims = {model.input_dims}")
    print(f"  model.output_dims = {model.output_dims}")
    
    print(f"\nTensor Network Structure:")
    for tensor in model.tn:
        print(f"  {tensor.tags}: shape={tensor.shape}, inds={tensor.inds}")
    
    assert model.n_features == n_features, f"Expected n_features={n_features}, got {model.n_features}"
    assert model.phys_dim == L + 1, f"Expected phys_dim={L+1}, got {model.phys_dim}"
    assert len(model.input_labels) == n_features, f"Expected {n_features} input labels"
    assert len(list(model.tn)) == n_features, f"Expected {n_features} tensors"
    
    print("\n[PASS] TNML_P structure verified!")
    return True


def test_tnml_f_structure():
    """Test TNML_F tensor network structure and bond dimensions."""
    print("\n" + "=" * 60)
    print("Testing TNML_F (Fourier)")
    print("=" * 60)
    
    n_features = 4
    L = 3
    bond_dim = 6
    output_dim = 2
    
    model = TNML_F(
        L=L,
        bond_dim=bond_dim,
        phys_dim=n_features,
        output_dim=output_dim,
        use_tn_normalization=False,
    )
    
    print(f"\nParameters: n_features={n_features}, L={L} (ignored), bond_dim={bond_dim}, output_dim={output_dim}")
    print(f"Expected: {n_features} nodes, each with phys_dim=2")
    print(f"\nModel attributes:")
    print(f"  model.n_features = {model.n_features}")
    print(f"  model.phys_dim = {model.phys_dim}")
    print(f"  model.bond_dim = {model.bond_dim}")
    print(f"  model.output_site = {model.output_site}")
    print(f"  model.input_labels = {model.input_labels}")
    print(f"  model.input_dims = {model.input_dims}")
    print(f"  model.output_dims = {model.output_dims}")
    
    print(f"\nTensor Network Structure:")
    for tensor in model.tn:
        print(f"  {tensor.tags}: shape={tensor.shape}, inds={tensor.inds}")
    
    assert model.n_features == n_features, f"Expected n_features={n_features}, got {model.n_features}"
    assert model.phys_dim == 2, f"Expected phys_dim=2, got {model.phys_dim}"
    assert len(model.input_labels) == n_features, f"Expected {n_features} input labels"
    assert len(list(model.tn)) == n_features, f"Expected {n_features} tensors"
    
    print("\n[PASS] TNML_F structure verified!")
    return True


def test_bond_dimension_matching():
    """Verify bond dimensions match between adjacent tensors."""
    print("\n" + "=" * 60)
    print("Testing Bond Dimension Matching")
    print("=" * 60)
    
    n_features = 5
    L = 2
    bond_dim = 8
    output_dim = 3
    
    for model_cls, name in [(TNML_P, "TNML_P"), (TNML_F, "TNML_F")]:
        print(f"\n--- {name} ---")
        model = model_cls(
            L=L,
            bond_dim=bond_dim,
            phys_dim=n_features,
            output_dim=output_dim,
            use_tn_normalization=False,
        )
        
        tensors = list(model.tn)
        for i, tensor in enumerate(tensors):
            for ind in tensor.inds:
                if ind.startswith('b'):
                    size = tensor.ind_size(ind)
                    print(f"  Node{i}, index {ind}: size={size}")
                    assert size == bond_dim, f"Bond dimension mismatch at {ind}: expected {bond_dim}, got {size}"
        
        print(f"  [PASS] All bond dimensions = {bond_dim}")
    
    return True


def test_single_feature():
    """Test models with a single feature (edge case)."""
    print("\n" + "=" * 60)
    print("Testing Single Feature (Edge Case)")
    print("=" * 60)
    
    for model_cls, name in [(TNML_P, "TNML_P"), (TNML_F, "TNML_F")]:
        print(f"\n--- {name} ---")
        model = model_cls(
            L=2,
            bond_dim=4,
            phys_dim=1,
            output_dim=2,
            use_tn_normalization=False,
        )
        
        tensors = list(model.tn)
        assert len(tensors) == 1, f"Expected 1 tensor, got {len(tensors)}"
        
        tensor = tensors[0]
        print(f"  Single tensor: shape={tensor.shape}, inds={tensor.inds}")
        assert "out" in tensor.inds, "Output index missing"
        assert "x0" in tensor.inds, "Input index missing"
        
        print(f"  [PASS] Single feature case works")
    
    return True


def test_comparison_with_mpo2():
    """Compare TNML structure with MPO2 to highlight differences."""
    print("\n" + "=" * 60)
    print("Comparison: TNML vs MPO2")
    print("=" * 60)
    
    from model.standard.MPO2_models import MPO2
    
    L = 3
    bond_dim = 6
    phys_dim = 5
    output_dim = 2
    
    mpo2 = MPO2(L=L, bond_dim=bond_dim, phys_dim=phys_dim, output_dim=output_dim, use_tn_normalization=False)
    tnml_p = TNML_P(L=L, bond_dim=bond_dim, phys_dim=phys_dim, output_dim=output_dim, use_tn_normalization=False)
    tnml_f = TNML_F(L=L, bond_dim=bond_dim, phys_dim=phys_dim, output_dim=output_dim, use_tn_normalization=False)
    
    print(f"\nWith L={L}, bond_dim={bond_dim}, phys_dim={phys_dim}:")
    print(f"\nMPO2:")
    print(f"  Number of nodes: {len(list(mpo2.tn))}")
    print(f"  Physical dim per node: {phys_dim}")
    print(f"  Input labels: {mpo2.input_labels}")
    
    print(f"\nTNML_P:")
    print(f"  Number of nodes: {len(list(tnml_p.tn))}")
    print(f"  Physical dim per node: {tnml_p.phys_dim} (L+1 = {L+1})")
    print(f"  Input labels: {tnml_p.input_labels}")
    
    print(f"\nTNML_F:")
    print(f"  Number of nodes: {len(list(tnml_f.tn))}")
    print(f"  Physical dim per node: {tnml_f.phys_dim}")
    print(f"  Input labels: {tnml_f.input_labels}")
    
    print(f"\nKey difference:")
    print(f"  MPO2: {L} nodes, each receives same input repeated (polynomial of degree L)")
    print(f"  TNML: {phys_dim} nodes (one per feature), different encoding per feature")
    
    return True


if __name__ == "__main__":
    test_tnml_p_structure()
    test_tnml_f_structure()
    test_bond_dimension_matching()
    test_single_feature()
    test_comparison_with_mpo2()
    
    print("\n" + "=" * 60)
    print("ALL TESTS PASSED!")
    print("=" * 60)
