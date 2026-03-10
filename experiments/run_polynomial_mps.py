# type: ignore
"""
Train MPS on 4th degree polynomial with noise and plot reconstruction.
"""

import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
torch.set_default_dtype(torch.float64)

from model.base.NTN import NTN
from model.losses import MSELoss
from model.utils import create_inputs, REGRESSION_METRICS
from model.standard import MPO2
from experiments.device_utils import DEVICE


def run_polynomial_mps(
    n_samples=200, degree=4, noise_std=0.05, L=4, bond_dim=8, n_epochs=5, seed=42
):
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Generate x in [-1, 1]
    x = torch.linspace(-1, 1, n_samples, dtype=torch.float64, device=DEVICE)

    # Polynomial from roots in [-1, 1]
    roots = [-0.9, -0.8, 0.6, 0.9]
    print(f"Roots: {roots}")

    y_clean = torch.ones_like(x)
    for r in roots:
        y_clean = y_clean * (x - r)

    # Add noise
    y_noisy = y_clean + torch.randn_like(x) * noise_std

    # Normalize
    x_mean, x_std = x.mean(), x.std()
    y_mean, y_std = y_noisy.mean(), y_noisy.std()
    x_norm = (x - x_mean) / x_std
    y_norm = (y_noisy - y_mean) / y_std

    # X = [x_norm, 1] for each sample
    X = torch.stack([x_norm, torch.ones_like(x_norm)], dim=1)
    y = y_norm.unsqueeze(1)

    # Model
    model = MPO2(L=L, bond_dim=bond_dim, phys_dim=2, output_dim=1)
    for t in model.tn:
        t.modify(data=t.data.to(DEVICE))

    # Train on ALL data
    loader = create_inputs(
        X, y, model.input_labels, model.output_dims, batch_size=64, append_bias=False
    )

    ntn = NTN(
        tn=model.tn,
        output_dims=model.output_dims,
        input_dims=model.input_dims,
        loss=MSELoss(),
        data_stream=loader,
    )
    ntn.fit(
        n_epochs=n_epochs, jitter=[0.01] * n_epochs, eval_metrics=REGRESSION_METRICS, verbose=True
    )

    # Forward on ALL data
    loader_all = create_inputs(
        X, y, model.input_labels, model.output_dims, batch_size=n_samples, append_bias=False
    )
    for batch in loader_all:
        inputs, _ = batch
        y_pred = ntn.forward(model.tn, [inputs])

    y_pred_denorm = (y_pred.data.cpu().squeeze() * y_std.cpu() + y_mean.cpu()).detach().numpy()

    # Plot
    x_np = x.cpu().numpy()
    y_clean_np = y_clean.cpu().numpy()
    y_noisy_np = y_noisy.cpu().numpy()

    plt.figure(figsize=(10, 6))
    plt.scatter(x_np, y_noisy_np, alpha=0.5, s=20, c="blue", label="Noisy data")
    plt.plot(x_np, y_clean_np, "g-", lw=2, label="True polynomial")
    plt.plot(x_np, y_pred_denorm, "r--", lw=2, label="MPS reconstruction")
    plt.xlabel("x")
    plt.ylabel("y")

    plt.legend()
    plt.grid(True, alpha=0.3)

    os.makedirs("results", exist_ok=True)
    plt.savefig("results/polynomial_mps_reconstruction.pdf", dpi=150)
    print(f"\nSaved: results/polynomial_mps_reconstruction.pdf")
    plt.show()


if __name__ == "__main__":
    run_polynomial_mps()
