#!/usr/bin/env python3
"""Test TNML models with NTN and GTN forward pass and environment computation."""

import sys
sys.path.insert(0, '.')

import torch
import torch.nn as nn
from model.standard.TNML import TNML_P, TNML_F
from model.utils import create_inputs_tnml, encode_polynomial, encode_fourier
from model.base.NTN import NTN
from model.base.GTN import GTN
from model.losses import MSELoss, CrossEntropyLoss


def test_tnml_p_forward():
    """Test TNML_P forward pass through NTN."""
    print("=" * 60)
    print("Testing TNML_P Forward Pass")
    print("=" * 60)
    
    n_samples = 16
    n_features = 4
    degree = 3
    bond_dim = 6
    output_dim = 2
    batch_size = 8
    
    X = torch.randn(n_samples, n_features)
    y = torch.randn(n_samples, output_dim)
    
    model = TNML_P(
        L=degree,
        bond_dim=bond_dim,
        phys_dim=n_features,
        output_dim=output_dim,
        use_tn_normalization=False,
        init_strength=0.1,
    )
    
    print(f"Model: TNML_P with {n_features} features, degree={degree}")
    print(f"  n_nodes: {len(list(model.tn))}")
    print(f"  phys_dim: {model.phys_dim}")
    print(f"  input_labels: {model.input_labels}")
    
    loader = create_inputs_tnml(
        X=X,
        y=y,
        input_labels=model.input_labels,
        output_labels=model.output_dims,
        batch_size=batch_size,
        encoding="polynomial",
        degree=degree,
    )
    
    print(f"\nLoader created with {len(loader)} batches")
    
    ntn = NTN(
        tn=model.tn,
        output_dims=model.output_dims,
        input_dims=model.input_dims,
        loss=MSELoss(),
        data_stream=loader,
    )
    
    print(f"NTN created successfully")
    
    mu, target = loader[0]
    print(f"\nFirst batch: {len(mu)} inputs, target shape: {target.shape}")
    
    output = ntn.forward(ntn.tn, loader.data_mu)
    print(f"Forward pass output shape: {output.shape}")
    print(f"Forward pass output:\n{output.data[:3]}")
    
    assert output.shape[1] == output_dim, f"Output dim mismatch"
    
    print("\n[PASS] TNML_P forward pass works!")
    return True


def test_tnml_f_forward():
    """Test TNML_F forward pass through NTN."""
    print("\n" + "=" * 60)
    print("Testing TNML_F Forward Pass")
    print("=" * 60)
    
    n_samples = 16
    n_features = 4
    bond_dim = 6
    output_dim = 2
    batch_size = 8
    
    X = torch.randn(n_samples, n_features)
    y = torch.randn(n_samples, output_dim)
    
    model = TNML_F(
        L=3,
        bond_dim=bond_dim,
        phys_dim=n_features,
        output_dim=output_dim,
        use_tn_normalization=False,
        init_strength=0.1,
    )
    
    print(f"Model: TNML_F with {n_features} features")
    print(f"  n_nodes: {len(list(model.tn))}")
    print(f"  phys_dim: {model.phys_dim}")
    print(f"  input_labels: {model.input_labels}")
    
    loader = create_inputs_tnml(
        X=X,
        y=y,
        input_labels=model.input_labels,
        output_labels=model.output_dims,
        batch_size=batch_size,
        encoding="fourier",
    )
    
    print(f"\nLoader created with {len(loader)} batches")
    
    ntn = NTN(
        tn=model.tn,
        output_dims=model.output_dims,
        input_dims=model.input_dims,
        loss=MSELoss(),
        data_stream=loader,
    )
    
    print(f"NTN created successfully")
    
    mu, target = loader[0]
    print(f"\nFirst batch: {len(mu)} inputs, target shape: {target.shape}")
    
    output = ntn.forward(ntn.tn, loader.data_mu)
    print(f"Forward pass output shape: {output.shape}")
    print(f"Forward pass output:\n{output.data[:3]}")
    
    assert output.shape[1] == output_dim, f"Output dim mismatch"
    
    print("\n[PASS] TNML_F forward pass works!")
    return True


def test_tnml_p_environment():
    """Test TNML_P environment computation (gradient calculation)."""
    print("\n" + "=" * 60)
    print("Testing TNML_P Environment Computation")
    print("=" * 60)
    
    n_samples = 16
    n_features = 4
    degree = 2
    bond_dim = 4
    output_dim = 2
    batch_size = 8
    
    X = torch.randn(n_samples, n_features)
    y = torch.randn(n_samples, output_dim)
    
    model = TNML_P(
        L=degree,
        bond_dim=bond_dim,
        phys_dim=n_features,
        output_dim=output_dim,
        use_tn_normalization=False,
        init_strength=0.1,
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
    
    ntn = NTN(
        tn=model.tn,
        output_dims=model.output_dims,
        input_dims=model.input_dims,
        loss=MSELoss(),
        data_stream=loader,
    )
    
    print(f"Testing environment for each node...")
    
    mu, target = loader[0]
    
    for i, tensor in enumerate(model.tn):
        tag = list(tensor.tags)[0]
        print(f"\n  Computing environment for {tag}...")
        
        env = ntn._batch_environment(mu, ntn.tn, tag)
        print(f"    Environment shape: {env.shape}")
        print(f"    Environment inds: {env.inds}")
        print(f"    Tensor inds: {tensor.inds}")
        
        tensor_inds = set(tensor.inds)
        env_inds = set(env.inds)
        out_labels = set(model.output_dims)
        
        expected_env_inds = ({'s'} | tensor_inds | out_labels) - (tensor_inds & out_labels)
        assert env_inds == expected_env_inds, f"Env inds mismatch: got {env_inds}, expected {expected_env_inds}"
        print(f"    [OK] Environment indices correct")
    
    print("\n[PASS] TNML_P environment computation works!")
    return True


def test_tnml_f_environment():
    """Test TNML_F environment computation (gradient calculation)."""
    print("\n" + "=" * 60)
    print("Testing TNML_F Environment Computation")
    print("=" * 60)
    
    n_samples = 16
    n_features = 4
    bond_dim = 4
    output_dim = 2
    batch_size = 8
    
    X = torch.randn(n_samples, n_features)
    y = torch.randn(n_samples, output_dim)
    
    model = TNML_F(
        L=3,
        bond_dim=bond_dim,
        phys_dim=n_features,
        output_dim=output_dim,
        use_tn_normalization=False,
        init_strength=0.1,
    )
    
    loader = create_inputs_tnml(
        X=X,
        y=y,
        input_labels=model.input_labels,
        output_labels=model.output_dims,
        batch_size=batch_size,
        encoding="fourier",
    )
    
    ntn = NTN(
        tn=model.tn,
        output_dims=model.output_dims,
        input_dims=model.input_dims,
        loss=MSELoss(),
        data_stream=loader,
    )
    
    print(f"Testing environment for each node...")
    
    mu, target = loader[0]
    
    for i, tensor in enumerate(model.tn):
        tag = list(tensor.tags)[0]
        print(f"\n  Computing environment for {tag}...")
        
        env = ntn._batch_environment(mu, ntn.tn, tag)
        print(f"    Environment shape: {env.shape}")
        print(f"    Environment inds: {env.inds}")
        print(f"    Tensor inds: {tensor.inds}")
        
        tensor_inds = set(tensor.inds)
        env_inds = set(env.inds)
        
        assert tensor_inds.issubset(env_inds), f"Tensor inds not in env: tensor={tensor_inds}, env={env_inds}"
        print(f"    [OK] Tensor indices present in environment")
    
    print("\n[PASS] TNML_F environment computation works!")
    return True


def test_tnml_p_single_update():
    """Test a single NTN update step for TNML_P."""
    print("\n" + "=" * 60)
    print("Testing TNML_P Single Update Step")
    print("=" * 60)
    
    n_samples = 32
    n_features = 4
    degree = 2
    bond_dim = 4
    output_dim = 2
    batch_size = 16
    
    X = torch.randn(n_samples, n_features)
    y = torch.randn(n_samples, output_dim)
    
    model = TNML_P(
        L=degree,
        bond_dim=bond_dim,
        phys_dim=n_features,
        output_dim=output_dim,
        use_tn_normalization=False,
        init_strength=0.1,
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
    
    ntn = NTN(
        tn=model.tn,
        output_dims=model.output_dims,
        input_dims=model.input_dims,
        loss=MSELoss(),
        data_stream=loader,
    )
    
    initial_loss = ntn.evaluate({'loss': lambda p, t: ((p - t) ** 2).sum() / t.numel()})['loss']
    print(f"Initial loss: {initial_loss:.6f}")
    
    tag = "Node0"
    print(f"\nUpdating {tag}...")
    ntn.update_tn_node(tag, regularize=True, jitter=1e-4)
    
    updated_loss = ntn.evaluate({'loss': lambda p, t: ((p - t) ** 2).sum() / t.numel()})['loss']
    print(f"Loss after update: {updated_loss:.6f}")
    
    print("\n[PASS] TNML_P single update step works!")
    return True


def test_tnml_f_single_update():
    """Test a single NTN update step for TNML_F."""
    print("\n" + "=" * 60)
    print("Testing TNML_F Single Update Step")
    print("=" * 60)
    
    n_samples = 32
    n_features = 4
    bond_dim = 4
    output_dim = 2
    batch_size = 16
    
    X = torch.randn(n_samples, n_features)
    y = torch.randn(n_samples, output_dim)
    
    model = TNML_F(
        L=3,
        bond_dim=bond_dim,
        phys_dim=n_features,
        output_dim=output_dim,
        use_tn_normalization=False,
        init_strength=0.1,
    )
    
    loader = create_inputs_tnml(
        X=X,
        y=y,
        input_labels=model.input_labels,
        output_labels=model.output_dims,
        batch_size=batch_size,
        encoding="fourier",
    )
    
    ntn = NTN(
        tn=model.tn,
        output_dims=model.output_dims,
        input_dims=model.input_dims,
        loss=MSELoss(),
        data_stream=loader,
    )
    
    initial_loss = ntn.evaluate({'loss': lambda p, t: ((p - t) ** 2).sum() / t.numel()})['loss']
    print(f"Initial loss: {initial_loss:.6f}")
    
    tag = "Node0"
    print(f"\nUpdating {tag}...")
    ntn.update_tn_node(tag, regularize=True, jitter=1e-4)
    
    updated_loss = ntn.evaluate({'loss': lambda p, t: ((p - t) ** 2).sum() / t.numel()})['loss']
    print(f"Loss after update: {updated_loss:.6f}")
    
    print("\n[PASS] TNML_F single update step works!")
    return True


def test_tnml_p_gtn_forward():
    """Test TNML_P forward pass through GTN."""
    print("\n" + "=" * 60)
    print("Testing TNML_P GTN Forward Pass")
    print("=" * 60)
    
    n_samples = 16
    n_features = 4
    degree = 3
    bond_dim = 6
    output_dim = 2
    batch_size = 8
    
    X = torch.randn(n_samples, n_features)
    y = torch.randn(n_samples, output_dim)
    
    model = TNML_P(
        L=degree,
        bond_dim=bond_dim,
        phys_dim=n_features,
        output_dim=output_dim,
        use_tn_normalization=False,
        init_strength=0.1,
    )
    
    print(f"Model: TNML_P with {n_features} features, degree={degree}")
    print(f"  n_nodes: {len(list(model.tn))}")
    print(f"  phys_dim: {model.phys_dim}")
    print(f"  input_dims: {model.input_dims}")
    
    gtn = GTN(
        tn=model.tn,
        output_dims=model.output_dims,
        input_dims=model.input_dims,
    )
    
    print(f"GTN created successfully")
    print(f"  Trainable params: {sum(p.numel() for p in gtn.parameters())}")
    
    X_encoded = encode_polynomial(X[:batch_size], degree)
    print(f"\nEncoded input shape: {X_encoded.shape}")
    
    X_list = [X_encoded[:, i, :] for i in range(n_features)]
    print(f"Input list: {len(X_list)} tensors, each shape {X_list[0].shape}")
    
    output = gtn(X_list)
    print(f"Forward pass output shape: {output.shape}")
    print(f"Forward pass output:\n{output[:3]}")
    
    assert output.shape == (batch_size, output_dim), f"Output shape mismatch"
    
    print("\n[PASS] TNML_P GTN forward pass works!")
    return True


def test_tnml_f_gtn_forward():
    """Test TNML_F forward pass through GTN."""
    print("\n" + "=" * 60)
    print("Testing TNML_F GTN Forward Pass")
    print("=" * 60)
    
    n_samples = 16
    n_features = 4
    bond_dim = 6
    output_dim = 2
    batch_size = 8
    
    X = torch.randn(n_samples, n_features)
    y = torch.randn(n_samples, output_dim)
    
    model = TNML_F(
        L=3,
        bond_dim=bond_dim,
        phys_dim=n_features,
        output_dim=output_dim,
        use_tn_normalization=False,
        init_strength=0.1,
    )
    
    print(f"Model: TNML_F with {n_features} features")
    print(f"  n_nodes: {len(list(model.tn))}")
    print(f"  phys_dim: {model.phys_dim}")
    print(f"  input_dims: {model.input_dims}")
    
    gtn = GTN(
        tn=model.tn,
        output_dims=model.output_dims,
        input_dims=model.input_dims,
    )
    
    print(f"GTN created successfully")
    print(f"  Trainable params: {sum(p.numel() for p in gtn.parameters())}")
    
    X_encoded = encode_fourier(X[:batch_size])
    print(f"\nEncoded input shape: {X_encoded.shape}")
    
    X_list = [X_encoded[:, i, :] for i in range(n_features)]
    print(f"Input list: {len(X_list)} tensors, each shape {X_list[0].shape}")
    
    output = gtn(X_list)
    print(f"Forward pass output shape: {output.shape}")
    print(f"Forward pass output:\n{output[:3]}")
    
    assert output.shape == (batch_size, output_dim), f"Output shape mismatch"
    
    print("\n[PASS] TNML_F GTN forward pass works!")
    return True


def test_tnml_p_gtn_training_step():
    """Test TNML_P GTN gradient descent step."""
    print("\n" + "=" * 60)
    print("Testing TNML_P GTN Training Step")
    print("=" * 60)
    
    n_samples = 32
    n_features = 4
    degree = 2
    bond_dim = 4
    output_dim = 2
    batch_size = 16
    
    X = torch.randn(n_samples, n_features)
    y = torch.randn(n_samples, output_dim)
    
    model = TNML_P(
        L=degree,
        bond_dim=bond_dim,
        phys_dim=n_features,
        output_dim=output_dim,
        use_tn_normalization=False,
        init_strength=0.1,
    )
    
    gtn = GTN(
        tn=model.tn,
        output_dims=model.output_dims,
        input_dims=model.input_dims,
    )
    
    optimizer = torch.optim.Adam(gtn.parameters(), lr=0.01)
    criterion = nn.MSELoss()
    
    X_encoded = encode_polynomial(X[:batch_size], degree)
    X_list = [X_encoded[:, i, :] for i in range(n_features)]
    y_batch = y[:batch_size]
    
    output = gtn(X_list)
    initial_loss = criterion(output, y_batch).item()
    print(f"Initial loss: {initial_loss:.6f}")
    
    for step in range(5):
        optimizer.zero_grad()
        output = gtn(X_list)
        loss = criterion(output, y_batch)
        loss.backward()
        optimizer.step()
    
    output = gtn(X_list)
    final_loss = criterion(output, y_batch).item()
    print(f"Loss after 5 steps: {final_loss:.6f}")
    print(f"Loss reduced: {initial_loss > final_loss}")
    
    print("\n[PASS] TNML_P GTN training step works!")
    return True


def test_tnml_f_gtn_training_step():
    """Test TNML_F GTN gradient descent step."""
    print("\n" + "=" * 60)
    print("Testing TNML_F GTN Training Step")
    print("=" * 60)
    
    n_samples = 32
    n_features = 4
    bond_dim = 4
    output_dim = 2
    batch_size = 16
    
    X = torch.randn(n_samples, n_features)
    y = torch.randn(n_samples, output_dim)
    
    model = TNML_F(
        L=3,
        bond_dim=bond_dim,
        phys_dim=n_features,
        output_dim=output_dim,
        use_tn_normalization=False,
        init_strength=0.1,
    )
    
    gtn = GTN(
        tn=model.tn,
        output_dims=model.output_dims,
        input_dims=model.input_dims,
    )
    
    optimizer = torch.optim.Adam(gtn.parameters(), lr=0.01)
    criterion = nn.MSELoss()
    
    X_encoded = encode_fourier(X[:batch_size])
    X_list = [X_encoded[:, i, :] for i in range(n_features)]
    y_batch = y[:batch_size]
    
    output = gtn(X_list)
    initial_loss = criterion(output, y_batch).item()
    print(f"Initial loss: {initial_loss:.6f}")
    
    for step in range(5):
        optimizer.zero_grad()
        output = gtn(X_list)
        loss = criterion(output, y_batch)
        loss.backward()
        optimizer.step()
    
    output = gtn(X_list)
    final_loss = criterion(output, y_batch).item()
    print(f"Loss after 5 steps: {final_loss:.6f}")
    print(f"Loss reduced: {initial_loss > final_loss}")
    
    print("\n[PASS] TNML_F GTN training step works!")
    return True


if __name__ == "__main__":
    print("=" * 60)
    print("NTN TESTS")
    print("=" * 60)
    test_tnml_p_forward()
    test_tnml_f_forward()
    test_tnml_p_environment()
    test_tnml_f_environment()
    test_tnml_p_single_update()
    test_tnml_f_single_update()
    
    print("\n" + "=" * 60)
    print("GTN TESTS")
    print("=" * 60)
    test_tnml_p_gtn_forward()
    test_tnml_f_gtn_forward()
    test_tnml_p_gtn_training_step()
    test_tnml_f_gtn_training_step()
    
    print("\n" + "=" * 60)
    print("ALL NTN AND GTN INTEGRATION TESTS PASSED!")
    print("=" * 60)
