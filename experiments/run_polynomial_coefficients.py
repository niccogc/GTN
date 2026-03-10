# type: ignore
"""
Train MPS on polynomial and extract/compare coefficients using CP contraction.

The key insight: D=[i,j,...] in find_s_simple specifies which monomial we want.
E.g., D=[1,1] with N=4 sites means x_1 * x_1 = x^2 coefficient.
     D=[1,1,1,1] means x^4, D=[] means constant term, etc.
"""

import os
import sys
import itertools
import torch
import numpy as np
import quimb.tensor as qt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
torch.set_default_dtype(torch.float64)

from model.base.NTN import NTN
from model.losses import MSELoss
from model.utils import create_inputs, REGRESSION_METRICS
from model.standard import MPO2
from experiments.device_utils import DEVICE


# ============ CP Construction (from combPython.py) ============


def base(dim, n):
    """Create a one-hot vector of size dim with 1 at position n."""
    A = np.zeros(dim)
    A[int(n)] = 1
    return A


def cp_vecs_from_coord(coord, dim):
    """Create list of one-hot vectors for each coordinate."""
    return [base(dim, i) for i in coord]


def cp_from_coord_list(coordinates, dim):
    """Stack CP vectors from list of coordinates.
    Returns array of shape (dim, n_coords, ...) for each site.
    """
    all_vec_sets = [cp_vecs_from_coord(c, dim) for c in coordinates]
    # all_vec_sets[i] is a list of N vectors (one per site) for coordinate i
    # We want to stack so that for each site we have all coordinate vectors
    stacked_vecs = np.stack(all_vec_sets, axis=1)  # (N_sites, n_coords, dim)
    return stacked_vecs


def find_s_simple(D, N):
    """Find all permutations for monomial degree D across N sites.

    D is a list of indices representing which "power" we want.
    E.g., D=[1,1] means x^2 (two x's), D=[1,1,1] means x^3, etc.

    For phys_dim=2: index 0 = constant (1), index 1 = x
    So D=[1,1] finds all ways to place two x's among N sites.
    """
    num_zeros = N - len(D)
    pool = list(D) + [0] * num_zeros
    return set(itertools.permutations(pool, N))


def extract_coefficient(mps_tn, D, N, phys_dim, debug=False):
    """Extract the coefficient for monomial specified by D from the MPS.
    
    Args:
        mps_tn: The MPS tensor network
        D: List specifying the monomial (e.g., [1,1] for x^2)
        N: Number of sites
        phys_dim: Physical dimension (2 for [1, x] embedding)
        debug: Print debug info

    Returns:
        The coefficient value
    """
    coords = find_s_simple(D, N)
    
    if debug:
        print(f"  D={D} -> coords from find_s_simple: {coords}")
    
    CP = cp_from_coord_list(list(coords), dim=phys_dim)
    
    if debug:
        print(f"  CP shape: {CP.shape}")
        print(f"  CP[0] (site 0 vectors):\n{CP[0]}")
    
    # Create tensor network for CP contraction
    # CP shape: (N_sites, n_coords, phys_dim)
    cp_tensors = []
    for site_idx in range(N):
        # For each site, create tensor with shape (phys_dim, n_coords)
        # that contracts with x{site_idx} index of MPS
        site_vecs = CP[site_idx]  # (n_coords, phys_dim)
        t = qt.Tensor(
            data=torch.tensor(site_vecs.T, dtype=torch.float64),  # (phys_dim, n_coords)
            inds=[f"x{site_idx}", "s"],
        )
        cp_tensors.append(t)

    # Contract MPS with CP tensors
    cp_tn = qt.TensorNetwork(cp_tensors)
    combined = mps_tn & cp_tn

    # Contract everything - keep 's' and 'out' as outer indices
    result = combined.contract(output_inds=['out', 's'])
    
    if debug:
        print(f"  result shape: {result.data.shape}")
        print(f"  result values: {result.data}")

    # Sum over the coordinate permutations (index "s")
    # This gives the total coefficient for this monomial
    coeff = result.data.sum(dim=-1) if len(result.data.shape) > 1 else result.data.sum()

    return coeff.item() if hasattr(coeff, "item") else float(coeff)


def run_polynomial_coefficients(n_samples=500, L=4, bond_dim=16, n_epochs=5, seed=42):
    """Train MPS on polynomial defined by coefficients and compare."""

    torch.manual_seed(seed)
    np.random.seed(seed)

    # ============ Define polynomial by coefficients ============
    # P(x) = c0 + c1*x + c2*x^2 + c3*x^3 + c4*x^4
    # For L=4 sites with phys_dim=2, we can represent up to degree 4

    true_coeffs = {
        0: 1.0,  # constant
        1: -2.0,  # x
        2: 0.5,  # x^2
        3: 1.5,  # x^3
        4: -1.0,  # x^4
    }

    print("=" * 60)
    print("TRUE POLYNOMIAL COEFFICIENTS:")
    print("=" * 60)
    for degree, coeff in true_coeffs.items():
        print(f"  x^{degree}: {coeff:+.4f}")
    print()

    # Generate x in [-1, 1]
    x = torch.linspace(-1, 1, n_samples, dtype=torch.float64, device=DEVICE)

    # Evaluate polynomial
    y_clean = torch.zeros_like(x)
    for degree, coeff in true_coeffs.items():
        y_clean = y_clean + coeff * (x**degree)

    # Add small noise
    noise_std = 0.01
    y_noisy = y_clean + torch.randn_like(x) * noise_std

    # Feature embedding: X = [1, x] for each sample
    # phys_dim=2: index 0 -> 1 (constant/bias), index 1 -> x
    X = torch.stack([torch.ones_like(x), x], dim=1)
    y = y_noisy.unsqueeze(1)

    # ============ Create and train MPS ============
    print("Training MPS...")
    print(f"  Sites (L): {L}")
    print(f"  Bond dimension: {bond_dim}")
    print(f"  Physical dimension: 2 (embedding [1, x])")
    print()

    model = MPO2(L=L, bond_dim=bond_dim, phys_dim=2, output_dim=1)
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

    # ============ Extract coefficients from trained MPS ============
    print("\n" + "=" * 60)
    print("EXTRACTING COEFFICIENTS FROM TRAINED MPS:")
    print("=" * 60)

    # For degree d, D has d ones: D=[1]*d
    # E.g., degree 0 -> D=[], degree 1 -> D=[1], degree 2 -> D=[1,1], etc.

    extracted_coeffs = {}
    for degree in range(L + 1):  # 0 to L inclusive
        D = [1] * degree
        print(f"\nExtracting degree {degree} (D={D}):")
        coeff = extract_coefficient(model.tn, D, L, phys_dim=2, debug=True)
        extracted_coeffs[degree] = coeff
        print(f"  -> coefficient = {coeff}")

    # ============ Compare coefficients ============
    print("\n" + "=" * 60)
    print("COEFFICIENT COMPARISON:")
    print("=" * 60)
    print(f"{'Degree':<8} {'True':<12} {'Extracted':<12} {'Error':<12}")
    print("-" * 44)

    for degree in range(L + 1):
        true_val = true_coeffs.get(degree, 0.0)
        ext_val = extracted_coeffs[degree]
        error = abs(true_val - ext_val)
        print(f"x^{degree:<6} {true_val:+.6f}    {ext_val:+.6f}    {error:.6f}")

    # ============ Verify with forward pass ============
    print("\n" + "=" * 60)
    print("FORWARD PASS VERIFICATION:")
    print("=" * 60)

    loader_all = create_inputs(
        X, y, model.input_labels, model.output_dims, batch_size=n_samples, append_bias=False
    )
    for batch in loader_all:
        inputs, _ = batch
        y_pred = ntn.forward(model.tn, [inputs])

    y_pred_np = y_pred.data.squeeze().detach().cpu().numpy()
    
    # Reconstruct polynomial from extracted coefficients
    x_np = x.cpu().numpy()
    y_from_extracted = np.zeros_like(x_np)
    for degree, coeff in extracted_coeffs.items():
        y_from_extracted += coeff * (x_np ** degree)
    
    # Reconstruct from true coefficients
    y_from_true = np.zeros_like(x_np)
    for degree, coeff in true_coeffs.items():
        y_from_true += coeff * (x_np ** degree)
    
    # Compare all three
    mse_pred_vs_true = ((y_pred_np - y_from_true) ** 2).mean()
    mse_extracted_vs_pred = ((y_from_extracted - y_pred_np) ** 2).mean()
    mse_extracted_vs_true = ((y_from_extracted - y_from_true) ** 2).mean()
    
    print(f"MSE (MPS prediction vs true poly):        {mse_pred_vs_true:.8f}")
    print(f"MSE (extracted coeffs vs MPS prediction): {mse_extracted_vs_pred:.8f}")
    print(f"MSE (extracted coeffs vs true poly):      {mse_extracted_vs_true:.8f}")
    
    # Sample comparison at a few x values
    print("\nSample comparison at x = -1, 0, 1:")
    print(f"{'x':<6} {'True':<12} {'MPS pred':<12} {'Extracted':<12}")
    print("-" * 42)
    for xi in [-1.0, 0.0, 1.0]:
        idx = np.argmin(np.abs(x_np - xi))
        print(f"{xi:<6.1f} {y_from_true[idx]:+.6f}    {y_pred_np[idx]:+.6f}    {y_from_extracted[idx]:+.6f}")

    return true_coeffs, extracted_coeffs


if __name__ == "__main__":
    run_polynomial_coefficients()
