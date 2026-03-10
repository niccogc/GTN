# type: ignore
"""
Train MPS on 2D polynomial surface defined by ROOTS.

f(x,y) = P(x) * Q(y) where P and Q are polynomials from roots.
This creates a separable polynomial with cross terms.
"""

import os
import sys
import itertools
import torch
import numpy as np
import quimb.tensor as qt
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
torch.set_default_dtype(torch.float64)

from model.base.NTN import NTN
from model.losses import MSELoss
from model.utils import create_inputs, REGRESSION_METRICS
from model.standard import MPO2
from experiments.device_utils import DEVICE


# ============ CP Construction ============


def base(dim, n):
    A = np.zeros(dim)
    A[int(n)] = 1
    return A


def cp_vecs_from_coord(coord, dim):
    return [base(dim, i) for i in coord]


def cp_from_coord_list(coordinates, dim):
    all_vec_sets = [cp_vecs_from_coord(c, dim) for c in coordinates]
    stacked_vecs = np.stack(all_vec_sets, axis=1)
    return stacked_vecs


def find_s_simple(D, N):
    num_zeros = N - len(D)
    pool = list(D) + [0] * num_zeros
    return set(itertools.permutations(pool, N))


def extract_coefficient(mps_tn, D, N, phys_dim):
    coords = find_s_simple(D, N)
    CP = cp_from_coord_list(list(coords), dim=phys_dim)

    cp_tensors = []
    for site_idx in range(N):
        site_vecs = CP[site_idx]
        t = qt.Tensor(
            data=torch.tensor(site_vecs.T, dtype=torch.float64),
            inds=[f"x{site_idx}", "s"],
        )
        cp_tensors.append(t)

    cp_tn = qt.TensorNetwork(cp_tensors)
    combined = mps_tn & cp_tn
    result = combined.contract(output_inds=["out", "s"])

    coeff = result.data.sum(dim=-1) if len(result.data.shape) > 1 else result.data.sum()
    return coeff.item() if hasattr(coeff, "item") else float(coeff)


def poly_from_roots(roots):
    """Get polynomial coefficients from roots.
    Returns coeffs [c0, c1, c2, ...] for c0 + c1*x + c2*x^2 + ...
    """
    # np.poly returns [highest, ..., lowest] so we reverse
    coeffs = np.poly(roots)[::-1]
    return coeffs


def run_surface_roots(n_points=350, L=4, bond_dim=4, n_epochs=3, seed=42):
    """Train MPS on 2D polynomial from roots and extract coefficients."""

    torch.manual_seed(seed)
    np.random.seed(seed)

    # ============ Define polynomial by ROOTS ============
    # P(x) = (x - r1)(x - r2)(x - r3)(x - r4)
    # Q(y) = (y - s1)(y - s2)(y - s3)(y - s4)
    # f(x,y) = P(x) * Q(y)

    x_roots = [-0.75, 0.25, -0.85, 0.8]# 4 roots for degree 4 in x
    y_roots = [-0.65, 0, 0.9, 0.2]  # close coupled roots in y

    print("=" * 60)
    print("POLYNOMIAL FROM ROOTS:")
    print("=" * 60)
    print(f"x roots: {x_roots}")
    print(f"y roots: {y_roots}")
    print()

    # Get coefficients for P(x) and Q(y)
    x_coeffs = poly_from_roots(x_roots)  # [c0, c1, c2, c3, c4]
    y_coeffs = poly_from_roots(y_roots)

    print("P(x) coefficients (from roots):")
    for i, c in enumerate(x_coeffs):
        print(f"  x^{i}: {c:+.6f}")

    print("\nQ(y) coefficients (from roots):")
    for i, c in enumerate(y_coeffs):
        print(f"  y^{i}: {c:+.6f}")

    # Compute full 2D polynomial coefficients: f(x,y) = P(x) * Q(y)
    # coeff of x^i * y^j = x_coeffs[i] * y_coeffs[j]
    true_coeffs = {}
    print("\nFull f(x,y) = P(x)*Q(y) coefficients:")
    print("-" * 50)
    for i, cx in enumerate(x_coeffs):
        for j, cy in enumerate(y_coeffs):
            true_coeffs[(i, j)] = cx * cy
            if abs(cx * cy) > 1e-10:
                term = f"x^{i}*y^{j}" if i > 0 and j > 0 else (f"x^{i}" if j == 0 else f"y^{j}")
                if i == 0 and j == 0:
                    term = "const"
                print(f"  {term:<12}: {cx * cy:+.6f}")
    print()

    # Generate grid
    x_lin = torch.linspace(-1, 1, n_points, dtype=torch.float64, device=DEVICE)
    y_lin = torch.linspace(-1, 1, n_points, dtype=torch.float64, device=DEVICE)
    xx, yy = torch.meshgrid(x_lin, y_lin, indexing="ij")
    x_flat = xx.flatten()
    y_flat = yy.flatten()
    n_samples = len(x_flat)

    # Evaluate polynomial: f(x,y) = P(x) * Q(y)
    # P(x) from roots
    px = torch.ones_like(x_flat)
    for r in x_roots:
        px = px * (x_flat - r)

    # Q(y) from roots
    qy = torch.ones_like(y_flat)
    for r in y_roots:
        qy = qy * (y_flat - r)

    z_clean = px * qy

    # Add small noise
    noise_std = 0.002
    z_noisy = z_clean + torch.randn_like(z_clean) * noise_std

    # Feature embedding: [1, x, y]
    X = torch.stack([torch.ones_like(x_flat), x_flat, y_flat], dim=1)
    y = z_noisy.unsqueeze(1)

    # ============ Train MPS ============
    print("Training MPS...")
    print(f"  Sites (L): {L}")
    print(f"  Bond dimension: {bond_dim}")
    print(f"  Physical dimension: 3 (embedding [1, x, y])")
    print(f"  Samples: {n_samples}")
    print()

    model = MPO2(L=L, bond_dim=bond_dim, phys_dim=3, output_dim=1)
    for t in model.tn:
        t.modify(data=t.data.to(DEVICE))

    loader = create_inputs(
        X, y, model.input_labels, model.output_dims, batch_size=350, append_bias=False
    )

    ntn = NTN(
        tn=model.tn,
        output_dims=model.output_dims,
        input_dims=model.input_dims,
        loss=MSELoss(),
        data_stream=loader,
    )

    ntn.fit(
        n_epochs=n_epochs, eval_metrics=REGRESSION_METRICS, jitter = 0.5, verbose=True
    )

    # ============ Extract coefficients ============
    print("\n" + "=" * 60)
    print("EXTRACTING COEFFICIENTS:")
    print("=" * 60)

    extracted_diagonal = {}
    extracted_cross = {}

    # ---- DIAGONAL TERMS ----
    print("\n--- DIAGONAL TERMS (x^n, y^n) ---")

    # Constant
    D = []
    coeff = extract_coefficient(model.tn, D, L, phys_dim=3)
    extracted_diagonal[(0, 0)] = coeff
    true_val = true_coeffs.get((0, 0), 0)
    print(f"  const:  {coeff:+.6f}  (true: {true_val:+.6f})  err: {abs(coeff - true_val):.6f}")

    # x^n
    for n in range(1, L + 1):
        D = [1] * n
        coeff = extract_coefficient(model.tn, D, L, phys_dim=3)
        extracted_diagonal[(n, 0)] = coeff
        true_val = true_coeffs.get((n, 0), 0)
        print(
            f"  x^{n}:    {coeff:+.6f}  (true: {true_val:+.6f})  err: {abs(coeff - true_val):.6f}"
        )

    # y^n
    for n in range(1, L + 1):
        D = [2] * n
        coeff = extract_coefficient(model.tn, D, L, phys_dim=3)
        extracted_diagonal[(0, n)] = coeff
        true_val = true_coeffs.get((0, n), 0)
        print(
            f"  y^{n}:    {coeff:+.6f}  (true: {true_val:+.6f})  err: {abs(coeff - true_val):.6f}"
        )

    # ---- CROSS TERMS ----
    print("\n--- CROSS TERMS (x^m * y^n) ---")

    for px in range(1, L):
        for py in range(1, L - px + 1):
            D = [1] * px + [2] * py
            coeff = extract_coefficient(model.tn, D, L, phys_dim=3)
            extracted_cross[(px, py)] = coeff
            true_val = true_coeffs.get((px, py), 0)
            print(
                f"  x^{px}*y^{py}: {coeff:+.6f}  (true: {true_val:+.6f})  err: {abs(coeff - true_val):.6f}"
            )

    # ============ Forward pass ============
    loader_all = create_inputs(
        X, y, model.input_labels, model.output_dims, batch_size=n_samples, append_bias=False
    )
    for batch in loader_all:
        inputs, _ = batch
        z_pred = ntn.forward(model.tn, [inputs])

    z_pred_np = z_pred.data.squeeze().detach().cpu().numpy()

    # ============ Reconstruct surfaces ============
    x_np = x_flat.cpu().numpy()
    y_np = y_flat.cpu().numpy()
    z_true_np = z_clean.cpu().numpy()

    # Diagonal only
    z_diagonal = np.zeros_like(x_np)
    for (px, py), coeff in extracted_diagonal.items():
        z_diagonal += coeff * (x_np**px) * (y_np**py)

    # Cross only
    z_cross = np.zeros_like(x_np)
    for (px, py), coeff in extracted_cross.items():
        z_cross += coeff * (x_np**px) * (y_np**py)

    # All extracted
    all_extracted = {**extracted_diagonal, **extracted_cross}
    z_all = np.zeros_like(x_np)
    for (px, py), coeff in all_extracted.items():
        z_all += coeff * (x_np**px) * (y_np**py)

    # ============ Plot ============
    print("\n" + "=" * 60)
    print("PLOTTING...")
    print("=" * 60)

    xx_np = xx.cpu().numpy()
    yy_np = yy.cpu().numpy()
    zz_true = z_true_np.reshape(n_points, n_points)
    zz_pred = z_pred_np.reshape(n_points, n_points)
    zz_diag = z_diagonal.reshape(n_points, n_points)
    zz_cross = z_cross.reshape(n_points, n_points)

    fig = plt.figure(figsize=(16, 12))

    ax1 = fig.add_subplot(221, projection="3d")
    ax1.plot_surface(xx_np, yy_np, zz_true, cmap="viridis", alpha=0.8)
    ax1.set_title(f"True: P(x)*Q(y)\nx_roots={x_roots}\ny_roots={y_roots}")
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax1.set_zlabel("z")

    ax2 = fig.add_subplot(222, projection="3d")
    ax2.plot_surface(xx_np, yy_np, zz_pred, cmap="plasma", alpha=0.8)
    ax2.set_title("MPS Reconstruction")
    ax2.set_xlabel("x")
    ax2.set_ylabel("y")
    ax2.set_zlabel("z")

    ax3 = fig.add_subplot(223, projection="3d")
    ax3.plot_surface(xx_np, yy_np, zz_diag, cmap="coolwarm", alpha=0.8)
    ax3.set_title("DIAGONAL Terms Only\n(x^n, y^n)")
    ax3.set_xlabel("x")
    ax3.set_ylabel("y")
    ax3.set_zlabel("z")

    ax4 = fig.add_subplot(224, projection="3d")
    ax4.plot_surface(xx_np, yy_np, zz_cross, cmap="inferno", alpha=0.8)
    ax4.set_title("CROSS Terms Only\n(x^m * y^n)")
    ax4.set_xlabel("x")
    ax4.set_ylabel("y")
    ax4.set_zlabel("z")

    plt.tight_layout()

    os.makedirs("results", exist_ok=True)
    plt.savefig("results/surface_roots.pdf", dpi=150)
    print(f"\nSaved: results/surface_roots.pdf")
    plt.show()

    # ============ MSE Summary ============
    mse_pred = ((z_pred_np - z_true_np) ** 2).mean()
    mse_diag = ((z_diagonal - z_true_np) ** 2).mean()
    mse_cross = ((z_cross - z_true_np) ** 2).mean()
    mse_all = ((z_all - z_true_np) ** 2).mean()
    mse_all_vs_mps = ((z_all - z_pred_np) ** 2).mean()

    print("\n" + "=" * 60)
    print("MSE COMPARISON:")
    print("=" * 60)
    print(f"MPS vs true:              {mse_pred:.8f}")
    print(f"Diagonal only vs true:    {mse_diag:.8f}")
    print(f"Cross only vs true:       {mse_cross:.8f}")
    print(f"All extracted vs true:    {mse_all:.8f}")
    print(f"All extracted vs MPS:     {mse_all_vs_mps:.8f}")

    return extracted_diagonal, extracted_cross, true_coeffs


if __name__ == "__main__":
    run_surface_roots()
