# BatchMovingEnvironment: Efficient Environment Caching for Batch Tensor Networks

## Overview

`BatchMovingEnvironment` is an extension of quimb's `MovingEnvironment` class that handles tensor networks with batch dimensions. It enables efficient optimization sweeps through 1D tensor networks (like MPS) while preserving batch indices for parallel computation.

## The Problem

In standard DMRG and tensor network optimization, we need to compute "environments" - the contraction of everything except a target site. For a site `i`, the environment looks like:

```
╭─●─●─●─●─●─╮     ╭─●─●─●─●─●─╮
│ │ │ │ │ │ │     │ │ │ │ │ │ │
H─H─H─H─H─H─H ... H─H─H─H─H─H─H
│ │ │ │ │ │ │     │ │ │ │ │ │ │
╰─●─●─●─●─●─╯     ╰─●─●─●─●─●─╯
    LEFT              RIGHT
          ^
       site i (excluded)
```

**Challenge**: When we have batch dimensions (e.g., processing 100 samples simultaneously), naive contraction would sum over the batch dimension, losing parallelism.

## The Solution

`BatchMovingEnvironment` extends quimb's `MovingEnvironment` by:

1. **Tracking batch indices**: Accepts `batch_inds` parameter (e.g., `['s']` for batch dimension)
2. **Excluding batch sites from environment**: Sites with batch indices are kept separate from the environment
3. **Efficient caching**: Pre-computes and caches environments for each position during sweep

## How It Works

### Initialization

```python
env = BatchMovingEnvironment(
    tn=tensor_network,    # Full TN (psi & phi & inputs)
    begin='left',         # Start from left
    bsz=1,               # Block size (sites to exclude)
    batch_inds=['s']     # Batch dimension indices
)
```

The initialization:
1. Identifies which sites contain batch indices
2. Sets up initial environments excluding these batch sites
3. Caches environments for efficient sweeping

### Environment Structure

At position `i`, the environment contains:
- All sites to the LEFT of position `i` (contracted)
- All sites to the RIGHT of position `i` (contracted)  
- **Excludes**: The target site at position `i` and any batch-dimension sites

Example for `L=4` sites, batch at site `i`:

```
Position 0:        Position 1:        Position 2:
╭─●─╮              ╭─●─●─╮            ╭─●─●─●─╮
│ │ │   ●─●─●      │ │ │ │   ●─●      │ │ │ │ │   ●
│ │ │   │ │ │      │ │ │ │   │ │      │ │ │ │ │   │
│ │ │---│─│─│      │ │ │ │---│─│      │ │ │ │ │---│
0 1 2   3         0 1 2 3      4      0 1 2 3 4
^target           ^target            ^target
```

### Usage Pattern

#### 1. Move to Target Site

```python
env.move_to(i)  # Move to site i
```

#### 2. Get Environment (with hole for target)

```python
current_env = env()  # Returns TN with all except current bsz sites
```

#### 3. Create Hole for Specific Target

```python
# Delete the specific target tensor to create the hole
env_with_hole = current_env.copy()
env_with_hole.delete(f"TARGET_TAG_{i}")
```

#### 4. Contract Environment (keeping batch dimension)

```python
# Get outer indices (bonds to target)
outer_inds = env_with_hole.outer_inds()

# Add batch index 's' to preserve it
if 's' not in outer_inds:
    inds_to_keep = list(outer_inds) + ['s']
else:
    inds_to_keep = outer_inds

# Contract environment
contracted_env = env_with_hole.contract(all, output_inds=inds_to_keep)
# Shape: (bond_dims..., batch_size)
```

#### 5. Use Environment for Updates

```python
# Compute gradient/update for target tensor
target = mps[f"I{i}"]

# The environment provides the gradient context
# contracted_env has shape matching target's bonds + batch dimension

# Update target (fake update example)
new_data = compute_update(target, contracted_env)
target.modify(data=new_data)
```

#### 6. Move to Next Site

```python
env.move_right()  # or env.move_left()
# Environment is recomputed incorporating the update
```

## Key Features

### 1. Efficient Caching

Environments are **pre-computed and cached**, not virtual views:
- ✅ Fast: Don't recompute everything at each site
- ✅ Memory efficient: Only store contracted environments
- ⚠️ Cached: After updating a tensor, must move to propagate changes

### 2. Batch Dimension Preservation

Unlike standard environment computation:
- Standard: Contracts everything → scalar or vector
- **BatchMovingEnvironment**: Preserves batch dimension → `(bond_dims..., batch_size)`

This enables:
- Parallel gradient computation for all batch samples
- Efficient Newton updates across batches
- Vectorized operations

### 3. Full Sweep Support

Supports bidirectional sweeps:

```python
# Forward sweep (left to right)
for i in range(L):
    env.move_to(i)
    # ... update site i ...

# Backward sweep (right to left)  
for i in reversed(range(L)):
    env.move_to(i)
    # ... update site i ...
```

## Example: Full Optimization Sweep

```python
# Setup
L = 10
batch_size = 100
mps = create_mps(L, batch_size)
inputs = create_inputs(L, batch_size)

tn = mps & inputs
env = BatchMovingEnvironment(tn, begin='left', bsz=1, batch_inds=['s'])

# Optimization sweep
for i in range(L):
    # Move to site
    env.move_to(i)
    
    # Get environment with hole
    current_env = env()
    env_with_hole = current_env.copy()
    env_with_hole.delete(f"MPS_BLOCK_{i}")
    
    # Contract keeping batch dimension
    outer_inds = list(env_with_hole.outer_inds())
    if 's' not in outer_inds:
        outer_inds.append('s')
    
    contracted_env = env_with_hole.contract(all, output_inds=outer_inds)
    
    # Compute update (e.g., Newton step)
    target = mps[f"I{i}"]
    gradient = compute_gradient(target, contracted_env)
    hessian = compute_hessian(target, contracted_env)
    update = solve_newton(gradient, hessian)
    
    # Apply update
    new_data = target.data - learning_rate * update
    target.modify(data=new_data)
    
    # Move incorporates the update into cached environments
```

## Implementation Details

### Inheritance from MovingEnvironment

```python
class BatchMovingEnvironment(qb.MovingEnvironment):
    def __init__(self, tn, begin, bsz, batch_inds=None, **kwargs):
        self.batch_inds = batch_inds or []
        # ... identify batch sites ...
        super().__init__(tn, begin, bsz, **kwargs)
```

### Batch Site Identification

The class identifies which sites contain batch indices by checking if any batch index appears in the site's tensor indices.

### Environment Initialization

Modifies the standard `init_segment` to exclude batch sites from environment calculation.

## Performance Benefits

For a tensor network with `L` sites and batch size `B`:

- **Without caching**: O(L × B × contraction_cost) per site
- **With BatchMovingEnvironment**: O(B × contraction_cost) per site (amortized)

The caching provides approximately **L×** speedup for multi-site sweeps!

## Usage in MPS_NTN

The `BatchMovingEnvironment` is designed for use in MPS_NTN (Matrix Product State Newton Tensor Networks):

```python
class MPS_NTN(NTN):
    def _batch_environment(self, batch_inds):
        """Compute environments for batch of samples."""
        tn = self._create_overlap_tn(batch_inds)
        env = BatchMovingEnvironment(
            tn, 
            begin='left', 
            bsz=1, 
            batch_inds=batch_inds
        )
        return env
    
    def _batch_forward(self, batch_inds):
        """Forward pass using cached environments."""
        env = self._batch_environment(batch_inds)
        
        for i in range(self.L):
            env.move_to(i)
            # ... compute local predictions ...
```

## Limitations and Notes

1. **Batch indices must be consistent**: All batch sites should use the same batch index names
2. **Updates require movement**: After modifying a tensor, must call `move_right()`/`move_left()` to propagate
3. **Memory vs Speed tradeoff**: Caching environments uses more memory but is much faster
4. **Works best with MPS structure**: Designed for 1D tensor networks with sequential structure

## References

- Original `MovingEnvironment`: quimb.tensor.tensor_dmrg.MovingEnvironment
- Used in: DMRG, TEBD, and other 1D tensor network algorithms
- Extended for: Batch processing in NTN (Newton Tensor Networks)
