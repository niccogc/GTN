"""
Debug (MPO)² matrix computation to find the issue.
"""
import torch
import quimb.tensor as qt
from model.MPS import MPO2_NTN, create_mpo2_tensors
from model.builder import Inputs
from model.losses import MSELoss
import numpy as np

torch.set_default_dtype(torch.float64)

print("="*80)
print("DEBUGGING (MPO)² MATRIX COMPUTATION")
print("="*80)

# Simple data
N_SAMPLES = 200
BATCH_SIZE = 50

x_raw = 2 * torch.rand(N_SAMPLES, 1) - 1
y_raw = x_raw**2 + 0.05 * torch.randn(N_SAMPLES, 1)
x_features = torch.cat([x_raw, torch.ones_like(x_raw)], dim=1)

print(f"\nData: {N_SAMPLES} samples, batch_size={BATCH_SIZE}")

# Create small (MPO)²
n_sites = 3
mps_bond = 3
mpo_bond = 3
hidden_dim = 2

print(f"\n(MPO)² Structure:")
print(f"  Sites: {n_sites}")
print(f"  MPS bond: {mps_bond}, MPO bond: {mpo_bond}")
print(f"  Hidden dim: {hidden_dim}")

mps_tensors, mpo_tensors = create_mpo2_tensors(
    n_sites=n_sites,
    mps_bond_dim=mps_bond,
    mpo_bond_dim=mpo_bond,
    lower_phys_dim=2,
    upper_phys_dim=hidden_dim,
    output_dim=1
)

loader = Inputs(
    inputs=[x_features],
    outputs=[y_raw],
    outputs_labels=["y"],
    input_labels=['x1', 'x2', 'x3'],
    batch_dim="batch",
    batch_size=BATCH_SIZE
)

model = MPO2_NTN.from_tensors(
    mps_tensors=mps_tensors,
    mpo_tensors=mpo_tensors,
    output_dims=["y"],
    input_dims=['x1', 'x2', 'x3'],
    loss=MSELoss(),
    data_stream=loader,
    method='cholesky'
)

print(f"\nNode grid: {model.node_grid}")

# Monkey-patch _get_node_update to add debugging
original_get_node_update = model._get_node_update

def debug_get_node_update(node_tag, regularize=True, jitter=1e-6):
    print(f"\n{'='*70}")
    print(f"UPDATING NODE: {node_tag} at position {model.tag_to_position.get(node_tag, 'unknown')}")
    print(f"  regularize={regularize}, jitter={jitter}")
    print('='*70)
    
    # Compute H and b
    b, H = model._compute_H_b(node_tag)
    
    print(f"\n1. Gradient (b) and Hessian (H) computed:")
    print(f"   b.inds: {b.inds}")
    print(f"   H.inds: {H.inds}")
    
    # Fuse indices
    variational_ind = b.inds
    map_H = {'rows': variational_ind, 'cols': [i + '_prime' for i in variational_ind]}
    map_b = {'cols': variational_ind}
    
    var_sizes = tuple(model.tn[node_tag].ind_size(i) for i in variational_ind)
    shape_map = {'cols': var_sizes}
    
    H.fuse(map_H, inplace=True)
    b.fuse(map_b, inplace=True)
    
    matrix_data = H.to_dense(['rows'], ['cols'])
    vector = -b.to_dense(['cols'])
    
    print(f"\n2. After fusing to matrix form:")
    print(f"   Matrix shape: {matrix_data.shape}")
    print(f"   Vector shape: {vector.shape}")
    print(f"   Total parameters: {matrix_data.shape[0]}")
    
    # Check matrix properties BEFORE regularization
    print(f"\n3. Matrix properties BEFORE regularization:")
    eigenvals = torch.linalg.eigvalsh(matrix_data)
    print(f"   Eigenvalue range: [{eigenvals.min():.6e}, {eigenvals.max():.6e}]")
    print(f"   Condition number: {(eigenvals.max() / eigenvals.min()).item():.6e}")
    print(f"   Negative eigenvalues: {(eigenvals < 0).sum().item()}")
    print(f"   Near-zero eigenvalues (<1e-10): {(eigenvals.abs() < 1e-10).sum().item()}")
    
    # Check if matrix is symmetric
    sym_error = torch.norm(matrix_data - matrix_data.T) / torch.norm(matrix_data)
    print(f"   Symmetry error: {sym_error:.6e}")
    
    if regularize:
        backend, lib = model.get_backend(matrix_data)
        
        # Get current weights
        current_node = model.tn[node_tag].copy()
        current_node.fuse(map_b, inplace=True)
        old_weight = current_node.to_dense(['cols'])
        
        print(f"\n4. Regularization (λ={jitter}):")
        print(f"   old_weight shape: {old_weight.shape}")
        print(f"   old_weight norm: {torch.norm(old_weight):.6e}")
        print(f"   old_weight range: [{old_weight.min():.6e}, {old_weight.max():.6e}]")
        
        # Add λI to H
        matrix_data.diagonal().add_(jitter)
        
        # Add λ * old_weight to b
        vector = vector + jitter * old_weight
        
        print(f"\n5. Matrix properties AFTER regularization:")
        eigenvals_reg = torch.linalg.eigvalsh(matrix_data)
        print(f"   Eigenvalue range: [{eigenvals_reg.min():.6e}, {eigenvals_reg.max():.6e}]")
        print(f"   Condition number: {(eigenvals_reg.max() / eigenvals_reg.min()).item():.6e}")
        print(f"   Minimum eigenvalue: {eigenvals_reg.min():.6e}")
        print(f"   All positive: {(eigenvals_reg > 0).all().item()}")
        
        # Check if still singular
        det_approx = torch.prod(eigenvals_reg)
        print(f"   Determinant (approx): {det_approx:.6e}")
        
    print(f"\n6. Attempting to solve linear system...")
    try:
        tensor_node_data = model.solve_linear_system(matrix_data, vector)
        print(f"   ✓ SUCCESS! Solution shape: {tensor_node_data.shape}")
        print(f"   Solution norm: {torch.norm(tensor_node_data):.6e}")
        print(f"   Solution range: [{tensor_node_data.min():.6e}, {tensor_node_data.max():.6e}]")
    except Exception as e:
        print(f"   ✗ FAILED: {str(e)}")
        print(f"\n   Diagnostics:")
        print(f"   - Matrix is {'symmetric' if sym_error < 1e-10 else 'NOT symmetric'}")
        print(f"   - Smallest eigenvalue: {eigenvals_reg.min():.6e}")
        print(f"   - Regularization may be insufficient")
        raise
    
    # Use original method to complete
    return original_get_node_update(node_tag, regularize, jitter)

model._get_node_update = debug_get_node_update

# Try training with different regularizations
print("\n" + "="*80)
print("ATTEMPTING TRAINING")
print("="*80)

for reg in [0.1, 1.0]:
    print(f"\n{'#'*80}")
    print(f"# Testing with λ = {reg}")
    print('#'*80)
    
    try:
        scores = model.fit(n_epochs=1, regularize=True, jitter=reg, verbose=True)
        print(f"\n✓ Epoch completed! MSE={scores['mse']:.6f}")
        break  # Success!
    except Exception as e:
        print(f"\n✗ Failed with λ={reg}")
        print(f"Error: {str(e)}")
        continue

print("\n" + "="*80)
print("DEBUG COMPLETE")
print("="*80)
