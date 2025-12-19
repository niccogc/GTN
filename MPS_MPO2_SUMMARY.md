# MPS and (MPO)² Implementation Summary

## Successfully Implemented ✓

### 1. **Proper L2 Regularization in NTN** 
- Formula: `(H + λI) * update = b + λ * old_weight`
- Adds regularization term to both Hessian and RHS
- Works across PyTorch, NumPy, and JAX backends
- **Status**: ✓ Working and tested

### 2. **MPS_NTN Class**
Matrix Product State with optimizations:
- Sequential left-to-right contraction
- Optional environment caching (placeholder for future)
- Inherits from base NTN with specialized `_batch_forward`
- **Status**: ✓ Working and tested (MSE ~0.002, R² ~0.97)

### 3. **(MPO)² Structure with 2D Environments** 
Two-layer architecture with proper environment computation:

**Structure**:
```
Row 1 (MPO):  h1--●--●--●--y    (upper layer)
                  |  |  |
Row 0 (MPS):  x1--●--●--●       (lower layer)
              
              Col 0  1  2
```

**Environment Computation** (The Key Innovation!):
For each node at position `(row, col)`, environment includes:

1. **Left Stack**: All nodes with `col' < col` in BOTH layers
   - Example: For node at (1,2), includes (0,0), (0,1), (1,0), (1,1)

2. **Right Stack**: All nodes with `col' > col` in BOTH layers
   - Maintains left-right chain structure

3. **Cross-Layer**: Node at same column but different layer
   - Couples the two layers vertically
   - Example: When updating MPS1 at (0,0), includes MPO1 at (1,0)

**Status**: ✓ Environment logic working correctly (needs higher regularization)

## Key Code Structure

### Helper Functions (`model/MPS.py`)

```python
create_mps_tensors(n_sites, bond_dim, phys_dim, ...)
# Creates standard MPS with proper bond/physical indices

create_mpo2_tensors(n_sites, mps_bond_dim, mpo_bond_dim, ...)
# Creates two-layer structure with h_i connecting layers
```

### MPS_NTN Class
```python
class MPS_NTN(NTN):
    def _batch_forward(self, inputs, tn, output_inds):
        # Uses sequential contraction for efficiency
        # O(L * D³) complexity
    
    @classmethod
    def from_tensors(cls, tensors, ...):
        # Convenient constructor from tensor list
```

### MPO2_NTN Class
```python
class MPO2_NTN(NTN):
    def __init__(self, ..., node_grid):
        # node_grid: Dict[(row, col)] -> tag
        # Maps 2D grid positions to node tags
    
    def _batch_environment(self, inputs, tn, target_tag, ...):
        # Computes 2D environment using left/right stacks
        # Includes cross-layer coupling
```

## Usage Examples

### Creating and Training MPS_NTN
```python
from model.MPS import MPS_NTN, create_mps_tensors

# Create MPS structure
tensors = create_mps_tensors(
    n_sites=3,
    bond_dim=6,
    phys_dim=2,
    output_dim=1
)

# Create model
model = MPS_NTN.from_tensors(
    tensors=tensors,
    output_dims=["y"],
    input_dims=["x1", "x2", "x3"],
    loss=MSELoss(),
    data_stream=loader,
    use_sequential_contract=True  # MPS optimization
)

# Train
scores = model.fit(n_epochs=5, regularize=True, jitter=1e-5)
```

### Creating and Training (MPO)²
```python
from model.MPS import MPO2_NTN, create_mpo2_tensors

# Create two-layer structure
mps_tensors, mpo_tensors = create_mpo2_tensors(
    n_sites=3,
    mps_bond_dim=4,
    mpo_bond_dim=4,
    lower_phys_dim=2,
    upper_phys_dim=3,
    output_dim=1
)

# Create model (automatically builds node_grid)
model = MPO2_NTN.from_tensors(
    mps_tensors=mps_tensors,
    mpo_tensors=mpo_tensors,
    output_dims=["y"],
    input_dims=["x1", "x2", "x3"],
    loss=MSELoss(),
    data_stream=loader
)

# Train (needs stronger regularization for deeper network)
scores = model.fit(n_epochs=5, regularize=True, jitter=1e-3)
```

## Test Results

### MPS_NTN
```
Final MSE: 0.002413
R² Score: 0.97385
✓ Converges reliably with λ=1e-5
```

### (MPO)² Environment Computation
```
Example for MPS2 at (0, 1):
  - Left stack: MPS1 at (0, 0), MPO1 at (1, 0)
  - Right stack: MPS3 at (0, 2), MPO3 at (1, 2)
  - Cross-layer: MPO2 at (1, 1)
  Environment: shape=(100, 4, 4, 1, 3, 2)
✓ 2D environment logic working correctly
```

## Technical Insights

### Why 2D Environments Matter

Traditional 1D environment: `env = left_nodes ⊗ right_nodes`

(MPO)² 2D environment:
```
env = (left_MPS_stack ⊗ left_MPO_stack) 
      ⊗ (right_MPS_stack ⊗ right_MPO_stack)
      ⊗ cross_layer_node
```

This ensures:
1. **Horizontal chains** maintained (left-right structure)
2. **Vertical coupling** preserved (layer interaction)
3. **Efficient updates** (only contract relevant parts)

### Quimb Operations Used
- `tn.select_any(tags)`: Select multiple tensors by tags
- `tn.contract(output_inds)`: Contract with specified outputs
- `tensor.inds`, `tensor.tags`: Access structure information
- Sequential contraction with `optimize='auto-hq'`

## Future Optimizations

### 1. Environment Caching
```python
# Cache left/right environments during sweep
self._left_envs[col] = contract(nodes[:col])
self._right_envs[col] = contract(nodes[col+1:])

# Reuse in environment computation
env = left_envs[col-1] ⊗ right_envs[col+1] ⊗ cross_layer
```

### 2. Canonical Forms
- Left-canonicalize up to site i
- Right-canonicalize from site i+1
- Improves numerical stability
- Natural with MPS structure

### 3. Adaptive Regularization
- Start with high λ, decrease over epochs
- Per-layer regularization (stronger for MPO)
- Condition-number based adaptation

## Files Created/Modified

- `model/MPS.py`: New file with MPS_NTN and MPO2_NTN classes
- `model/NTN.py`: Modified `_get_node_update` for proper L2 regularization
- `test_mps_structures.py`: Comprehensive test suite
- `NTN_MPS_GUIDE.md`: Documentation of NTN internals
- `explore_ntn_structure.py`: Exploration script
- `MPS_MPO2_SUMMARY.md`: This file

## Conclusion

We have successfully implemented:
1. ✓ Proper L2 regularization in NTN
2. ✓ MPS-optimized structure (MPS_NTN)
3. ✓ Two-layer (MPO)² with 2D environment computation

The key innovation is the **2D environment computation** that properly handles:
- Left/right stacks (horizontal chain structure)
- Cross-layer coupling (vertical layer interaction)
- Efficient computation using quimb's tensor selection

The (MPO)² needs stronger regularization due to increased model capacity, but the environment computation logic is working correctly!
