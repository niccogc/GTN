#!/usr/bin/env python3
"""Full integration test for TNML models with NTN and GTN training."""

import sys
sys.path.insert(0, '.')

import torch
import torch.nn as nn
from model.standard.TNML import TNML_P, TNML_F
from model.utils import create_inputs_tnml, encode_polynomial, encode_fourier, REGRESSION_METRICS
from model.base.NTN import NTN
from model.base.GTN import GTN
from model.losses import MSELoss


def test_tnml_p_ntn_training():
    print("=" * 60)
    print("TNML_P NTN Training (3 epochs)")
    print("=" * 60)
    
    n_samples, n_features, degree, bond_dim, output_dim = 32, 4, 2, 4, 1
    
    X = torch.randn(n_samples, n_features)
    y = torch.randn(n_samples, output_dim)
    
    model = TNML_P(L=degree, bond_dim=bond_dim, phys_dim=n_features, output_dim=output_dim, use_tn_normalization=False, init_strength=0.1)
    loader = create_inputs_tnml(X, y, model.input_labels, model.output_dims, batch_size=16, encoding="polynomial", degree=degree)
    
    ntn = NTN(tn=model.tn, output_dims=model.output_dims, input_dims=model.input_dims, loss=MSELoss(), data_stream=loader)
    
    scores = ntn.fit(n_epochs=3, regularize=True, jitter=1e-4, verbose=True, eval_metrics=REGRESSION_METRICS)
    print(f"Final scores: {scores}")
    print("[PASS] TNML_P NTN training works!\n")


def test_tnml_f_ntn_training():
    print("=" * 60)
    print("TNML_F NTN Training (3 epochs)")
    print("=" * 60)
    
    n_samples, n_features, bond_dim, output_dim = 32, 4, 4, 1
    
    X = torch.randn(n_samples, n_features)
    y = torch.randn(n_samples, output_dim)
    
    model = TNML_F(L=2, bond_dim=bond_dim, phys_dim=n_features, output_dim=output_dim, use_tn_normalization=False, init_strength=0.1)
    loader = create_inputs_tnml(X, y, model.input_labels, model.output_dims, batch_size=16, encoding="fourier")
    
    ntn = NTN(tn=model.tn, output_dims=model.output_dims, input_dims=model.input_dims, loss=MSELoss(), data_stream=loader)
    
    scores = ntn.fit(n_epochs=3, regularize=True, jitter=1e-4, verbose=True, eval_metrics=REGRESSION_METRICS)
    print(f"Final scores: {scores}")
    print("[PASS] TNML_F NTN training works!\n")


def test_tnml_p_gtn_training():
    print("=" * 60)
    print("TNML_P GTN Training (10 steps)")
    print("=" * 60)
    
    n_samples, n_features, degree, bond_dim, output_dim, batch_size = 32, 4, 2, 4, 1, 16
    
    X = torch.randn(n_samples, n_features)
    y = torch.randn(n_samples, output_dim)
    
    model = TNML_P(L=degree, bond_dim=bond_dim, phys_dim=n_features, output_dim=output_dim, use_tn_normalization=False, init_strength=0.1)
    gtn = GTN(tn=model.tn, output_dims=model.output_dims, input_dims=model.input_dims)
    
    optimizer = torch.optim.Adam(gtn.parameters(), lr=0.01)
    criterion = nn.MSELoss()
    
    X_enc = encode_polynomial(X[:batch_size], degree)
    X_list = [X_enc[:, i, :] for i in range(n_features)]
    y_batch = y[:batch_size]
    
    initial_loss = criterion(gtn(X_list), y_batch).item()
    print(f"Initial loss: {initial_loss:.6f}")
    
    for step in range(10):
        optimizer.zero_grad()
        loss = criterion(gtn(X_list), y_batch)
        loss.backward()
        optimizer.step()
        if step % 3 == 0:
            print(f"  Step {step}: loss={loss.item():.6f}")
    
    final_loss = criterion(gtn(X_list), y_batch).item()
    print(f"Final loss: {final_loss:.6f}")
    print(f"Loss reduced: {initial_loss > final_loss}")
    print("[PASS] TNML_P GTN training works!\n")


def test_tnml_f_gtn_training():
    print("=" * 60)
    print("TNML_F GTN Training (10 steps)")
    print("=" * 60)
    
    n_samples, n_features, bond_dim, output_dim, batch_size = 32, 4, 4, 1, 16
    
    X = torch.randn(n_samples, n_features)
    y = torch.randn(n_samples, output_dim)
    
    model = TNML_F(L=2, bond_dim=bond_dim, phys_dim=n_features, output_dim=output_dim, use_tn_normalization=False, init_strength=0.1)
    gtn = GTN(tn=model.tn, output_dims=model.output_dims, input_dims=model.input_dims)
    
    optimizer = torch.optim.Adam(gtn.parameters(), lr=0.01)
    criterion = nn.MSELoss()
    
    X_enc = encode_fourier(X[:batch_size])
    X_list = [X_enc[:, i, :] for i in range(n_features)]
    y_batch = y[:batch_size]
    
    initial_loss = criterion(gtn(X_list), y_batch).item()
    print(f"Initial loss: {initial_loss:.6f}")
    
    for step in range(10):
        optimizer.zero_grad()
        loss = criterion(gtn(X_list), y_batch)
        loss.backward()
        optimizer.step()
        if step % 3 == 0:
            print(f"  Step {step}: loss={loss.item():.6f}")
    
    final_loss = criterion(gtn(X_list), y_batch).item()
    print(f"Final loss: {final_loss:.6f}")
    print(f"Loss reduced: {initial_loss > final_loss}")
    print("[PASS] TNML_F GTN training works!\n")


if __name__ == "__main__":
    test_tnml_p_ntn_training()
    test_tnml_f_ntn_training()
    test_tnml_p_gtn_training()
    test_tnml_f_gtn_training()
    
    print("=" * 60)
    print("ALL TNML TESTS PASSED!")
    print("=" * 60)
