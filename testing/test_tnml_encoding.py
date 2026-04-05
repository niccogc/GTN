#!/usr/bin/env python3
"""Test TNML encoding functions."""

import sys
sys.path.insert(0, '.')

import torch
import math
from model.utils import encode_polynomial, encode_fourier, create_inputs_tnml
from model.standard.TNML import TNML_P, TNML_F


def test_encode_polynomial():
    """Test polynomial encoding."""
    print("=" * 60)
    print("Testing encode_polynomial")
    print("=" * 60)
    
    X = torch.tensor([[0.5, 1.0], [2.0, 0.0]])
    degree = 3
    
    result = encode_polynomial(X, degree)
    
    print(f"Input X shape: {X.shape}")
    print(f"Input X:\n{X}")
    print(f"\nDegree: {degree}")
    print(f"Output shape: {result.shape}")
    print(f"Expected shape: (2, 2, 4)")
    
    assert result.shape == (2, 2, degree + 1), f"Shape mismatch: {result.shape}"
    
    expected_feature0_sample0 = torch.tensor([1.0, 0.5, 0.25, 0.125])
    expected_feature1_sample0 = torch.tensor([1.0, 1.0, 1.0, 1.0])
    expected_feature0_sample1 = torch.tensor([1.0, 2.0, 4.0, 8.0])
    
    print(f"\nSample 0, Feature 0: {result[0, 0]}")
    print(f"Expected: {expected_feature0_sample0}")
    assert torch.allclose(result[0, 0], expected_feature0_sample0), "Mismatch at sample 0, feature 0"
    
    print(f"\nSample 0, Feature 1: {result[0, 1]}")
    print(f"Expected: {expected_feature1_sample0}")
    assert torch.allclose(result[0, 1], expected_feature1_sample0), "Mismatch at sample 0, feature 1"
    
    print(f"\nSample 1, Feature 0: {result[1, 0]}")
    print(f"Expected: {expected_feature0_sample1}")
    assert torch.allclose(result[1, 0], expected_feature0_sample1), "Mismatch at sample 1, feature 0"
    
    print("\n[PASS] Polynomial encoding verified!")
    return True


def test_encode_fourier():
    """Test Fourier encoding."""
    print("\n" + "=" * 60)
    print("Testing encode_fourier")
    print("=" * 60)
    
    X = torch.tensor([[0.0, 1.0], [0.5, 2.0]])
    
    result = encode_fourier(X)
    
    print(f"Input X shape: {X.shape}")
    print(f"Input X:\n{X}")
    print(f"\nOutput shape: {result.shape}")
    print(f"Expected shape: (2, 2, 2)")
    
    assert result.shape == (2, 2, 2), f"Shape mismatch: {result.shape}"
    
    pi_half = math.pi / 2
    expected_00 = torch.tensor([math.cos(0.0 * pi_half), math.sin(0.0 * pi_half)])
    expected_01 = torch.tensor([math.cos(1.0 * pi_half), math.sin(1.0 * pi_half)])
    expected_10 = torch.tensor([math.cos(0.5 * pi_half), math.sin(0.5 * pi_half)])
    
    print(f"\nSample 0, Feature 0: {result[0, 0]}")
    print(f"Expected [cos(0), sin(0)] = {expected_00}")
    assert torch.allclose(result[0, 0], expected_00, atol=1e-6), "Mismatch"
    
    print(f"\nSample 0, Feature 1: {result[0, 1]}")
    print(f"Expected [cos(pi/2), sin(pi/2)] = {expected_01}")
    assert torch.allclose(result[0, 1], expected_01, atol=1e-6), "Mismatch"
    
    print(f"\nSample 1, Feature 0: {result[1, 0]}")
    print(f"Expected [cos(pi/4), sin(pi/4)] = {expected_10}")
    assert torch.allclose(result[1, 0], expected_10, atol=1e-6), "Mismatch"
    
    print("\n[PASS] Fourier encoding verified!")
    return True


def test_create_inputs_tnml_polynomial():
    """Test creating inputs for TNML_P model."""
    print("\n" + "=" * 60)
    print("Testing create_inputs_tnml (polynomial)")
    print("=" * 60)
    
    n_samples = 10
    n_features = 4
    degree = 3
    output_dim = 2
    batch_size = 5
    
    X = torch.randn(n_samples, n_features)
    y = torch.randn(n_samples, output_dim)
    
    model = TNML_P(
        L=degree,
        bond_dim=6,
        phys_dim=n_features,
        output_dim=output_dim,
        use_tn_normalization=False,
    )
    
    loader = create_inputs_tnml(
        X=X,
        y=y,
        input_labels=model.input_labels,
        output_labels=model.output_dims,
        batch_size=batch_size,
        encoding="polynomial",
        degree=degree,
    )
    
    print(f"Model input_labels: {model.input_labels}")
    print(f"Model phys_dim: {model.phys_dim}")
    print(f"Loader: {loader}")
    
    mu, target = loader[0]
    print(f"\nFirst batch:")
    print(f"  Number of input tensors: {len(mu)}")
    print(f"  Target shape: {target.shape}")
    
    for i, t in enumerate(mu):
        print(f"  Input {i}: shape={t.shape}, inds={t.inds}")
        expected_phys = degree + 1
        assert t.shape[1] == expected_phys, f"Expected phys_dim={expected_phys}, got {t.shape[1]}"
    
    assert len(mu) == n_features, f"Expected {n_features} inputs, got {len(mu)}"
    
    print("\n[PASS] create_inputs_tnml (polynomial) verified!")
    return True


def test_create_inputs_tnml_fourier():
    """Test creating inputs for TNML_F model."""
    print("\n" + "=" * 60)
    print("Testing create_inputs_tnml (Fourier)")
    print("=" * 60)
    
    n_samples = 10
    n_features = 4
    output_dim = 2
    batch_size = 5
    
    X = torch.randn(n_samples, n_features)
    y = torch.randn(n_samples, output_dim)
    
    model = TNML_F(
        L=3,
        bond_dim=6,
        phys_dim=n_features,
        output_dim=output_dim,
        use_tn_normalization=False,
    )
    
    loader = create_inputs_tnml(
        X=X,
        y=y,
        input_labels=model.input_labels,
        output_labels=model.output_dims,
        batch_size=batch_size,
        encoding="fourier",
    )
    
    print(f"Model input_labels: {model.input_labels}")
    print(f"Model phys_dim: {model.phys_dim}")
    print(f"Loader: {loader}")
    
    mu, target = loader[0]
    print(f"\nFirst batch:")
    print(f"  Number of input tensors: {len(mu)}")
    print(f"  Target shape: {target.shape}")
    
    for i, t in enumerate(mu):
        print(f"  Input {i}: shape={t.shape}, inds={t.inds}")
        assert t.shape[1] == 2, f"Expected phys_dim=2 for Fourier, got {t.shape[1]}"
    
    assert len(mu) == n_features, f"Expected {n_features} inputs, got {len(mu)}"
    
    print("\n[PASS] create_inputs_tnml (Fourier) verified!")
    return True


def test_dimension_matching():
    """Test that input dimensions match model expectations."""
    print("\n" + "=" * 60)
    print("Testing Dimension Matching (Model <-> Inputs)")
    print("=" * 60)
    
    n_features = 5
    degree = 2
    
    model_p = TNML_P(L=degree, bond_dim=4, phys_dim=n_features, output_dim=1, use_tn_normalization=False)
    model_f = TNML_F(L=degree, bond_dim=4, phys_dim=n_features, output_dim=1, use_tn_normalization=False)
    
    X = torch.randn(8, n_features)
    y = torch.randn(8, 1)
    
    loader_p = create_inputs_tnml(X, y, model_p.input_labels, encoding="polynomial", degree=degree)
    loader_f = create_inputs_tnml(X, y, model_f.input_labels, encoding="fourier")
    
    mu_p, _ = loader_p[0]
    mu_f, _ = loader_f[0]
    
    print("TNML_P:")
    for tensor in model_p.tn:
        for ind in tensor.inds:
            if ind.startswith('x'):
                model_size = tensor.ind_size(ind)
                print(f"  Model tensor {tensor.tags}: index {ind} has size {model_size}")
    
    for i, inp in enumerate(mu_p):
        inp_size = inp.shape[1]
        print(f"  Input tensor {i}: index x{i} has size {inp_size}")
        assert inp_size == model_p.phys_dim, f"Dimension mismatch at x{i}"
    
    print("\nTNML_F:")
    for tensor in model_f.tn:
        for ind in tensor.inds:
            if ind.startswith('x'):
                model_size = tensor.ind_size(ind)
                print(f"  Model tensor {tensor.tags}: index {ind} has size {model_size}")
    
    for i, inp in enumerate(mu_f):
        inp_size = inp.shape[1]
        print(f"  Input tensor {i}: index x{i} has size {inp_size}")
        assert inp_size == model_f.phys_dim, f"Dimension mismatch at x{i}"
    
    print("\n[PASS] All dimensions match!")
    return True


if __name__ == "__main__":
    test_encode_polynomial()
    test_encode_fourier()
    test_create_inputs_tnml_polynomial()
    test_create_inputs_tnml_fourier()
    test_dimension_matching()
    
    print("\n" + "=" * 60)
    print("ALL ENCODING TESTS PASSED!")
    print("=" * 60)
