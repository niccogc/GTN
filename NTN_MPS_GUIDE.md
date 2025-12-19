# NTN and MPS Structure Guide

## Overview
This document explains how NTN (Newton Tensor Networks) works with MPS (Matrix Product States) structures, including the key methods `batch_forward` and `batch_environment`.

## MPS Structure

### Tensor Indices
For an MPS with 3 sites:

```
Node1: (physical, right_bond)  ->  inds=('x1', 'b1'),  shape=(2, 3)
Node2: (left_bond, physical, right_bond, output) -> inds=('b1', 'x2', 'b2', 'y'), shape=(3, 2, 3, 1)
Node3: (left_bond, physical)  ->  inds=('b2', 'x3'),  shape=(3, 2)
```

- **Physical indices** (`x1`, `x2`, `x3`): Connect to input data
- **Bond indices** (`b1`, `b2`): Connect MPS tensors together (internal)
- **Output indices** (`y`): The prediction output

### Quimb MPS Conventions
From quimb docs:
- `shape='lrp'` means left bond, right bond, physical index
- `left_inds` and `right_inds` used for splitting/tracing operations
- End tensors drop either 'l' or 'r' from the shape

## Key NTN Methods

### 1. `_batch_forward(inputs, tn, output_inds)`

**Purpose**: Perform forward pass to get predictions

**Input**:
- `inputs`: List of input tensors, each with `inds=[batch_dim, physical_ind]`
  - Example: `[Tensor(inds=('batch', 'x1'), shape=(50, 2)), ...]`
- `tn`: The tensor network (MPS structure)
- `output_inds`: List of indices to keep in output (e.g., `['batch', 'y']`)

**Output**:
- Result tensor with `inds=output_inds`
- Example: `Tensor(inds=('batch', 'y'), shape=(50, 1))`

**How it works**:
```python
full_tn = tn & inputs  # Combine TN with inputs
res = full_tn.contract(output_inds=output_inds)  # Contract everything
res.transpose_(*output_inds)  # Reorder indices
```

### 2. `_batch_environment(inputs, tn, target_tag, sum_over_batch, sum_over_output)`

**Purpose**: Compute the "environment" around a target node (everything except that node)

**Input**:
- `inputs`: Same as batch_forward
- `tn`: Tensor network
- `target_tag`: Tag of the node to exclude (e.g., 'Node2')
- `sum_over_batch`: Whether to sum over the batch dimension
- `sum_over_output`: Whether to sum over output dimensions

**Output**:
- Environment tensor with indices:
  - `batch_dim` (if not summed)
  - `output_dims` (if not summed)  
  - **Bond indices that connect to target node**

**Example** for Node2:
```
Environment inds: ('batch', 'x3', 'b1', 'b2')
                    │       │     │    │
                    │       │     └────┴──── Bond indices connecting to Node2
                    │       └───────────────  Remaining physical index
                    └───────────────────────  Batch dimension

Target Node2 inds: ('b1', 'x2', 'b2', 'y')
                     │           │
                     └───────────┴────────── Matches bond indices in environment!
```

**Different modes**:
- `sum_over_batch=False, sum_over_output=False`: Full environment (batch & output)
  - `inds: ('batch', 'x3', 'b1', 'b2'), shape: (50, 2, 3, 3)`
- `sum_over_batch=True, sum_over_output=False`: Sum batch, keep output
  - `inds: ('x3', 'b1', 'b2'), shape: (2, 3, 3)`
- `sum_over_batch=True, sum_over_output=True`: Sum both
  - `inds: ('x3', 'b1', 'b2'), shape: (2, 3, 3)`

### 3. `_batch_node_derivatives(inputs, y_true, node_tag)`

**Purpose**: Compute gradient and Hessian for a specific node

**Steps**:
1. Compute environment `E` (excludes target node)
2. Reconstruct prediction: `y_pred = E ⊗ target_node`
3. Compute loss derivatives: `dL/dy`, `d²L/dy²`
4. Compute **gradient**: `G = E ⊗ dL/dy`
5. Compute **Hessian**: `H = E ⊗ d²L/dy² ⊗ E'`

**Output**:
- `node_grad`: Tensor with same indices as target node
- `node_hess`: Tensor with indices = `node_inds + node_inds_prime`

## MPS Optimization Opportunities

### 1. Sequential Contraction
For MPS, contract left-to-right:
```
input[0]--●--●--●--...--●--output
          |  |  |       |
```
Cost: `O(L * D³)` where L = length, D = bond dimension

### 2. Cached Environments (DMRG-style)
Pre-compute and cache:
- **Left environments**: Everything to the left of node i
- **Right environments**: Everything to the right of node i

Then: `env[i] = left_env[i-1] ⊗ right_env[i+1]`

### 3. Canonical Forms
Put MPS in left/right canonical form for numerical stability:
- **Left canonical**: All tensors to the left are orthogonal
- **Right canonical**: All tensors to the right are orthogonal
- **Mixed canonical**: Orthogonality center at specific site

## Creating an MPS-NTN Subclass

```python
class MPS_NTN(NTN):
    def _batch_forward(self, inputs, tn, output_inds):
        # Use sequential left-to-right contraction
        result_tn = tn.copy()
        for inp in inputs:
            result_tn = result_tn & inp
        return result_tn.contract(output_inds=output_inds, optimize='auto-hq')
    
    def _batch_environment(self, inputs, tn, target_tag, ...):
        # Can add caching for left/right environments here
        # For now, use parent implementation
        return super()._batch_environment(inputs, tn, target_tag, ...)
```

## Common Patterns

### Input Tensor Creation
```python
# For each physical index, create input tensor
input_tensors = []
for label in ['x1', 'x2', 'x3']:
    inp = qt.Tensor(data, inds=['batch', label], tags=f'Input_{label}')
    input_tensors.append(inp)
```

### Contracting Environment with Node
```python
# Get environment (without target node)
env = model._batch_environment(inputs, tn, 'Node2', ...)

# Get target node
target = tn['Node2']

# Predict: contract environment with target
y_pred = (env & target).contract(output_inds=['batch', 'y'])
```

## Key Insights

1. **Bond indices are crucial**: They define how environment connects to target node
2. **Index order matters**: Use `transpose_` to ensure consistent ordering
3. **Batch dimension**: Usually kept throughout, summed only when computing total gradients/Hessians
4. **Output dimensions**: Summed when computing node updates (we only care about effect on parameters)

## Regularization Implementation

Proper L2 regularization in `_get_node_update`:
```python
# H and b are the Hessian and gradient (fused to matrix/vector form)
if regularize:
    # Get current node weights
    old_weight = tn[node_tag].fuse(...).to_dense()
    
    # Add λI to H
    H.diagonal().add_(lambda_reg)
    
    # Add λ * old_weight to RHS
    b = b + lambda_reg * old_weight
    
# Solve: (H + λI) * update = b + λ * old_weight
update = solve(H, b)
```
