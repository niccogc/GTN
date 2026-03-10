# type: ignore
"""
Train MPS on 2D polynomial surface and extract diagonal coefficients.

Embedding: [1, x, y] at each site (phys_dim=3)
- index 0 = 1 (constant)
- index 1 = x
- index 2 = y

Diagonal terms: pure powers x^n, y^n (no cross terms xy, x²y, etc.)
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
    """Find all permutations for monomial specified by D across N sites."""
    num_zeros = N - len(D)
    pool = list(D) + [0] * num_zeros
    return set(itertools.permutations(pool, N))


def extract_coefficient(mps_tn, D, N, phys_dim, debug=False):
    """Extract coefficient for monomial D from MPS."""
    coords = find_s_simple(D, N)

    if debug:
        print(f"  D={D} -> coords: {coords}")

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


def run_surface_coefficients(n_points=20, L=4, bond_dim=16, n_epochs=10, seed=42):
    """Train MPS on 2D polynomial surface and extract diagonal coefficients."""

    torch.manual_seed(seed)
    np.random.seed(seed)

    # ============ Define 2D polynomial ============
    # f(x,y) = c00 + c10*x + c01*y + c20*x² + c02*y² + c11*xy + ...
    # We'll define diagonal terms (pure powers) + some cross terms

    true_coeffs = {
        # (x_power, y_power): coefficient
        (0, 0): 1.0,  # constant
        (1, 0): -1.5,  # x
        (0, 1): 2.0,  # y
        (2, 0): 0.5,  # x²
        (0, 2): -0.8,  # y²
        (1, 1): 0.3,  # xy (cross term)
        (3, 0): 0.2,  # x³
        (0, 3): -0.3,  # y³
        (4, 0): -0.1,  # x⁴
        (0, 4): 0.15,  # y⁴
    }

    print("=" * 60)
    print("TRUE POLYNOMIAL COEFFICIENTS:")
    print("=" * 60)
    for (px, py), coeff in sorted(true_coeffs.items()):
        term = f"x^{px}*y^{py}" if px > 0 and py > 0 else (f"x^{px}" if py == 0 else f"y^{py}")
        if px == 0 and py == 0:
            term = "const"
        print(f"  {term}: {coeff:+.4f}")
    print()

    # Generate grid
    x_lin = torch.linspace(-1, 1, n_points, dtype=torch.float64, device=DEVICE)
    y_lin = torch.linspace(-1, 1, n_points, dtype=torch.float64, device=DEVICE)
    xx, yy = torch.meshgrid(x_lin, y_lin, indexing="ij")
    x_flat = xx.flatten()
    y_flat = yy.flatten()
    n_samples = len(x_flat)

    # Evaluate polynomial
    z_clean = torch.zeros_like(x_flat)
    for (px, py), coeff in true_coeffs.items():
        z_clean = z_clean + coeff * (x_flat**px) * (y_flat**py)

    # Add small noise
    noise_std = 0.01
    z_noisy = z_clean + torch.randn_like(z_clean) * noise_std

    # Feature embedding: [1, x, y] for each sample
    # phys_dim=3: index 0 -> 1, index 1 -> x, index 2 -> y
    X = torch.stack([torch.ones_like(x_flat), x_flat, y_flat], dim=1)
    y = z_noisy.unsqueeze(1)

    # ============ Create and train MPS ============
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

    # ============ Extract diagonal coefficients ============
    print("\n" + "=" * 60)
    print("EXTRACTING DIAGONAL COEFFICIENTS:")
    print("=" * 60)
    
    # For phys_dim=3: index 0=1, index 1=x, index 2=y
    # D=[1]*n = x^n, D=[2]*n = y^n
    # D=[1]*px + [2]*py = x^px * y^py (cross terms)
    
    extracted_diagonal = {}
    extracted_cross = {}
    
    # ---- DIAGONAL TERMS ----
    print("\n--- DIAGONAL TERMS (x^n, y^n) ---")
    
    # Constant term
    D = []
    coeff = extract_coefficient(model.tn, D, L, phys_dim=3)
    extracted_diagonal[(0, 0)] = coeff
    print(f"  const:  {coeff:+.6f}  (true: {true_coeffs.get((0, 0), 0):+.6f})")
    
    # x^n terms
    for n in range(1, L + 1):
        D = [1] * n
        coeff = extract_coefficient(model.tn, D, L, phys_dim=3)
        extracted_diagonal[(n, 0)] = coeff
        true_val = true_coeffs.get((n, 0), 0)
        print(f"  x^{n}:    {coeff:+.6f}  (true: {true_val:+.6f})")
    
    # y^n terms
    for n in range(1, L + 1):
        D = [2] * n
        coeff = extract_coefficient(model.tn, D, L, phys_dim=3)
        extracted_diagonal[(0, n)] = coeff
        true_val = true_coeffs.get((0, n), 0)
        print(f"  y^{n}:    {coeff:+.6f}  (true: {true_val:+.6f})")
    
    # ---- CROSS TERMS ----
    print("\n--- CROSS TERMS (x^m * y^n) ---")
    
    # All combinations where px >= 1, py >= 1, px + py <= L
    for px in range(1, L):
        for py in range(1, L - px + 1):
            D = [1] * px + [2] * py
            coeff = extract_coefficient(model.tn, D, L, phys_dim=3)
            extracted_cross[(px, py)] = coeff
            true_val = true_coeffs.get((px, py), 0)
            print(f"  x^{px}*y^{py}: {coeff:+.6f}  (true: {true_val:+.6f})")
    # ============ Forward pass for MPS reconstruction ============
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

    # True surface
    z_true_np = z_clean.cpu().numpy()

    # Surface from diagonal coefficients only
    z_diagonal = np.zeros_like(x_np)
    for (px, py), coeff in extracted_diagonal.items():
        z_diagonal += coeff * (x_np**px) * (y_np**py)

    # ============ Plot ============
    print("\n" + "=" * 60)
    print("PLOTTING...")
    print("=" * 60)

    # Reshape for plotting
    xx_np = xx.cpu().numpy()
    yy_np = yy.cpu().numpy()
    zz_true = z_true_np.reshape(n_points, n_points)
    zz_pred = z_pred_np.reshape(n_points, n_points)
    zz_diag = z_diagonal.reshape(n_points, n_points)
    
    # Surface from ONLY cross terms
    z_cross = np.zeros_like(x_np)
    for (px, py), coeff in extracted_cross.items():
        z_cross += coeff * (x_np ** px) * (y_np ** py)
    zz_cross = z_cross.reshape(n_points, n_points)
    
    # Combine all extracted coefficients
    all_extracted = {**extracted_diagonal, **extracted_cross}
    z_all_extracted = np.zeros_like(x_np)
    for (px, py), coeff in all_extracted.items():
        z_all_extracted += coeff * (x_np ** px) * (y_np ** py)

    fig = plt.figure(figsize=(16, 12))

    # Plot 1: True surface
    ax1 = fig.add_subplot(221, projection="3d")
    ax1.plot_surface(xx_np, yy_np, zz_true, cmap="viridis", alpha=0.8)
    ax1.set_title("True Polynomial Surface")
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax1.set_zlabel("z")

    # Plot 2: MPS reconstruction
    ax2 = fig.add_subplot(222, projection="3d")
    ax2.plot_surface(xx_np, yy_np, zz_pred, cmap="plasma", alpha=0.8)
    ax2.set_title("MPS Reconstruction")
    ax2.set_xlabel("x")
    ax2.set_ylabel("y")
    ax2.set_zlabel("z")

    # Plot 3: Diagonal terms ONLY
    ax3 = fig.add_subplot(223, projection="3d")
    ax3.plot_surface(xx_np, yy_np, zz_diag, cmap="coolwarm", alpha=0.8)
    ax3.set_title("DIAGONAL Terms Only\n(x^n, y^n)")
    ax3.set_xlabel("x")
    ax3.set_ylabel("y")
    ax3.set_zlabel("z")

    # Plot 4: Cross terms ONLY
    ax4 = fig.add_subplot(224, projection="3d")
    ax4.plot_surface(xx_np, yy_np, zz_cross, cmap="inferno", alpha=0.8)
    ax4.set_title("CROSS Terms Only\n(x^m * y^n)")
    ax4.set_xlabel("x")
    ax4.set_ylabel("y")
    ax4.set_zlabel("z")

    plt.tight_layout()

    os.makedirs("results", exist_ok=True)
    plt.savefig("results/surface_coefficients.pdf", dpi=150)
    print(f"\nSaved: results/surface_coefficients.pdf")
    plt.show()

    # ============ Summary ============
    print("\n" + "=" * 60)
    print("SUMMARY - DIAGONAL COEFFICIENTS:")
    print("=" * 60)
    print(f"{'Term':<10} {'True':<12} {'Extracted':<12} {'Error':<12}")
    print("-" * 46)

    for px, py in sorted(extracted_diagonal.keys()):
        true_val = true_coeffs.get((px, py), 0)
        ext_val = extracted_diagonal[(px, py)]
        error = abs(true_val - ext_val)
        if px == 0 and py == 0:
            term = "const"
        elif py == 0:
            term = f"x^{px}"
        else:
            term = f"y^{py}"
        print(f"{term:<10} {true_val:+.6f}    {ext_val:+.6f}    {error:.6f}")
    
    print("\n" + "=" * 60)
    print("SUMMARY - CROSS TERMS:")
    print("=" * 60)
    print(f"{'Term':<10} {'True':<12} {'Extracted':<12} {'Error':<12}")
    print("-" * 46)
    
    for px, py in sorted(extracted_cross.keys()):
        true_val = true_coeffs.get((px, py), 0)
        ext_val = extracted_cross[(px, py)]
        error = abs(true_val - ext_val)
        term = f"x^{px}*y^{py}"
        print(f"{term:<10} {true_val:+.6f}    {ext_val:+.6f}    {error:.6f}")
    
    mse_pred = ((z_pred_np - z_true_np) ** 2).mean()
    mse_diag = ((z_diagonal - z_true_np) ** 2).mean()
    mse_all = ((z_all_extracted - z_true_np) ** 2).mean()
    mse_extracted_vs_mps = ((z_all_extracted - z_pred_np) ** 2).mean()
    
    print("\n" + "=" * 60)
    print("MSE COMPARISON:")
    print("=" * 60)
    print(f"MPS vs true:                    {mse_pred:.8f}")
    print(f"Diagonal only vs true:          {mse_diag:.8f}")
    print(f"All extracted vs true:          {mse_all:.8f}")
    print(f"All extracted vs MPS:           {mse_extracted_vs_mps:.8f}")

    return extracted_diagonal, extracted_cross, true_coeffs


if __name__ == "__main__":
    run_surface_coefficients()
