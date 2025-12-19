# Environment-Based Forward Pass Optimization

## Overview

This document explains the **computational advantage** of using pre-computed environments for forward passes in Neural Tensor Networks (NTN), particularly for MPS structures.

## The Key Insight

### Standard Forward Pass

```python
# Contract ALL nodes + inputs every time
full_tn = tn & inputs
y_pred = full_tn.contract(output_inds=[batch_dim, output_dims])
```

**Cost**: Full tensor network contraction every time you need predictions.

### Environment-Based Forward Pass

```python
# 1. Compute environment ONCE (everything except target node)
env = _batch_environment(inputs, tn, target_tag='Node2')

# 2. FAST forward passes with different node values
y_pred_1 = (env & node_value_1).contract(...)  # Cheap!
y_pred_2 = (env & node_value_2).contract(...)  # Cheap!
y_pred_3 = (env & node_value_3).contract(...)  # Cheap!
```

**Advantage**: Once environment is computed, testing different node values is **much faster** because you only need to contract the environment with the new node, not the entire network.

## Already Implemented in NTN!

The code **already** uses this optimization in `_batch_node_derivatives` (model/NTN.py:176-181):

```python
# 1. Calculate Environment E 
env = self._batch_environment(inputs, tn, target_tag=node_tag, ...)

# 2. Reconstruct y_pred from Environment + Current Node
# This avoids running a full forward pass separately
target_tensor = tn[node_tag]
y_pred = (env & target_tensor).contract(output_inds=[batch_dim, output_dims])
```

## New API: Making It Accessible

### 1. `forward_from_environment()` (General NTN)

**Use case**: Fast forward pass once environment is computed

```python
# Compute environment once
env = model.get_environment(
    model.tn, 'Node2', input_gen, 
    sum_over_batch=False, sum_over_output=False
)

# Test multiple node values efficiently
for candidate_value in search_space:
    new_node = qt.Tensor(candidate_value, inds=model.tn['Node2'].inds, 
                        tags={'Node2'})
    y_pred = model.forward_from_environment(env, 'Node2', node_tensor=new_node)
    # Much faster than calling model.forward() each time!
```

**Speedup**: Avoid re-contracting the entire network for each candidate.

### 2. `forward_with_updated_node()` (MPS_NTN)

**Use case**: Quickly test an updated node value

```python
# Test if a Newton step improves loss
old_node_data = model.tn['Node2'].data
new_node_data = old_node_data + newton_step

y_pred_new = model.forward_with_updated_node(inputs, 'Node2', new_node_data)

if loss(y_pred_new, y_true) < loss(y_pred_old, y_true):
    model.tn['Node2'].modify(data=new_node_data)  # Accept update
```

**Speedup**: Environment is cached or computed once, then reused.

## Memory Tradeoffs

### The Problem: Batch Dimension Makes Caching Expensive

**Environment size**: `O(batch_size * bond_dim^2 * num_features)`

Example for MPS with 3 sites, bond_dim=6, batch_size=100:
```
Environment for Node2: shape (100, 6, 6, 2)
Memory: 100 * 6 * 6 * 2 * 8 bytes = 57.6 KB per environment
```

**For entire sweep**: `O(num_nodes * batch_size * bond_dim^2)` 

With 10 nodes: ~576 KB per batch

### When to Use Caching

✅ **Good use cases:**
- Small batch sizes (< 100 samples)
- Line search / parameter testing (many forward passes with same inputs)
- Validation during training (repeated evaluation on same batch)
- Interactive optimization

❌ **Bad use cases:**
- Large batch sizes (> 500 samples)  
- Single forward pass per batch
- Memory-constrained systems

### Quimb's MovingEnvironment

Quimb has a built-in `MovingEnvironment` class for DMRG-style sweeps:

```python
from quimb.tensor.tensor_dmrg import MovingEnvironment

# For overlap tensor network (ket & bra)
overlap_tn = ket.H & ket  
env = MovingEnvironment(
    tn=overlap_tn,
    begin='left',  # Start from left
    bsz=2,         # 2-site DMRG
    cyclic=False
)

# Move environment to site i
env.move_to(i)

# Get current environment (everything except sites i, i+1)
current_env = env()
```

**Problem for NTN**: MovingEnvironment is designed for **overlap calculations** (ket & bra), not for **forward passes with inputs**. Our environments need to include:
- Input tensors with batch dimensions
- Output dimensions
- Open bonds connecting to target node

So we use our custom `_batch_environment` method instead.

## Implementation Details

### NTN.forward_from_environment()

**Location**: model/NTN.py (after `forward` method)

**Signature**:
```python
def forward_from_environment(self, env, node_tag, node_tensor=None, 
                            sum_over_batch=False) -> qt.Tensor
```

**Parameters**:
- `env`: Pre-computed environment (from `get_environment` or `_batch_environment`)
- `node_tag`: Tag of the node that was excluded from environment
- `node_tensor`: Optional custom node tensor. If None, uses current node from self.tn
- `sum_over_batch`: Whether to sum over batch dimension in output

**Returns**: Predictions y_pred

### MPS_NTN Additional Methods

**Location**: model/MPS.py (after `_batch_forward` method)

#### `clear_environment_cache()`
Frees memory by clearing cached environments.

#### `get_cached_environment(inputs, node_tag)`
Get environment from cache or compute and cache it.

**Warning**: Caching with batch dimension is memory-heavy! Only use when:
1. Batch size is small (< 100)
2. Many forward passes needed with same inputs
3. Memory is not a constraint

#### `forward_with_updated_node(inputs, node_tag, new_node_data)`
Efficiently compute forward pass with a temporarily updated node.

**Main use case**: Line search, parameter testing, validation.

## Practical Recommendations

### For General Tensor Networks

```python
# Scenario: Testing multiple parameter values

# Option 1: Without caching (memory-efficient)
for candidate in candidates:
    # Compute environment each time (still faster than full forward)
    env = model.get_environment(model.tn, node_tag, inputs, ...)
    y_pred = model.forward_from_environment(env, node_tag, 
                                           node_tensor=candidate)
```

### For MPS Structures

```python
# Scenario: Newton's method with line search

# Enable caching for this optimization run
model_mps = MPS_NTN(..., cache_environments=True)

for epoch in range(n_epochs):
    for node_tag in trainable_nodes:
        # Compute environment (cached)
        env = model_mps.get_cached_environment(inputs, node_tag)
        
        # Compute gradient and Hessian (uses environment internally)
        delta = model_mps._get_node_update(node_tag, ...)
        
        # Line search: test multiple step sizes
        for alpha in [1.0, 0.5, 0.25, 0.1]:
            new_value = old_value + alpha * delta
            y_pred = model_mps.forward_from_environment(env, node_tag, 
                                                       qt.Tensor(new_value, ...))
            if loss(y_pred) < best_loss:
                best_alpha = alpha
        
        # Apply best update
        model_mps.tn[node_tag].modify(data=old_value + best_alpha * delta)
    
    # Clear cache after epoch to free memory
    model_mps.clear_environment_cache()
```

### For MPO² Structures

**Challenge**: Two-layer structure makes environments even larger.

**Recommendation**: 
- Don't use caching (too memory-intensive)
- Use `forward_from_environment` without caching
- Consider diagonal Hessian approximation to reduce memory

## Performance Benchmarks (TODO)

We need to create benchmarks measuring:

1. **Forward pass speedup**:
   - Standard forward vs environment-based forward
   - With/without caching

2. **Memory usage**:
   - Cache size vs batch size
   - Cache size vs bond dimension

3. **Training speedup**:
   - With line search (multiple forward passes per update)
   - Without line search (single forward pass per update)

## Summary

| Feature | Memory Cost | Speed Gain | When to Use |
|---------|-------------|------------|-------------|
| `forward_from_environment` | None (no cache) | Moderate | Testing multiple node values with same inputs |
| `get_cached_environment` | O(batch * bond²) | High | Small batches, repeated forward passes |
| Quimb MovingEnvironment | Similar | High | DMRG overlap calculations (not for NTN forward) |

**Key Insight**: The computational advantage is **only useful when you need multiple forward passes with the same inputs but different node values**. For standard training (one forward pass per batch), the environment-based approach is already used internally in `_batch_node_derivatives` and doesn't need explicit caching.

**Best Practice**: Use `forward_from_environment` when needed, but **don't cache** unless you have a specific reason (line search, validation, parameter sweep) and memory allows it.
