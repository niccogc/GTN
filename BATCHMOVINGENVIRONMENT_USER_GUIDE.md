# BatchMovingEnvironment User Guide

## Overview

`BatchMovingEnvironment` is a specialized environment manager for tensor network sweeps that preserves batch and output dimensions. It's built on top of `quimb.MovingEnvironment` and is designed for efficient DMRG-style optimization with batched data.

## Table of Contents

1. [When to Use BatchMovingEnvironment](#when-to-use)
2. [Initialization](#initialization)
3. [Environment Structure](#environment-structure)
4. [Creating Holes](#creating-holes)
5. [Movement Operations](#movement-operations)
6. [Complete Example](#complete-example)
7. [Advanced Usage](#advanced-usage)
8. [Troubleshooting](#troubleshooting)

---

## When to Use

Use `BatchMovingEnvironment` when:
- You have a tensor network with **batch dimensions** (e.g., batch index `'s'`)
- You have **output dimensions** (e.g., output index `'out'`)
- You need to perform **sweeping optimizations** (left-to-right, right-to-left)
- You want **efficient caching** of left/right environments

**Example Use Cases:**
- Matrix Product State (MPS) optimization with batched inputs
- Newton Tensor Network (NTN) training
- DMRG with multiple samples

---

## Initialization

### Step 1: Prepare Your Tensor Network

```python
import quimb.tensor as qt
import torch

# Create an MPS
L = 4  # Number of sites
psi = qt.MPS_rand_state(L, bond_dim=3, phys_dim=2)

# Convert to torch (if needed)
psi.apply_to_arrays(lambda x: torch.tensor(x, dtype=torch.float32))

# Reindex physical dimensions for clarity
psi.reindex({f"k{i}": f"phys_{i}" for i in range(L)}, inplace=True)

# Add unique tags to each site (CRITICAL!)
for i in range(L):
    psi.add_tag(f"MPS_{i}", where=f"I{i}")
```

### Step 2: Add Batch Inputs

```python
BATCH_SIZE = 100

inputs = []
for i in range(L):
    inp = qt.Tensor(
        data=torch.rand(BATCH_SIZE, 2, dtype=torch.float32),
        inds=['s', f'phys_{i}'],  # 's' is the batch index
        tags={f'I{i}', 'INPUT', f'INPUT_{i}'}
    )
    inputs.append(inp)

# Build full tensor network
tn = psi.copy()
for inp in inputs:
    tn.add_tensor(inp)
```

### Step 3: (Optional) Add Output Dimension

```python
OUTPUT_DIM = 10
middle_site = 2

# Add output dimension BEFORE converting to torch
middle_tensor = psi[f'I{middle_site}']
middle_tensor.new_ind('out', size=OUTPUT_DIM, axis=-1)
```

### Step 4: Initialize BatchMovingEnvironment

```python
from batch_moving_environment import BatchMovingEnvironment

env = BatchMovingEnvironment(
    tn,                      # Your tensor network
    begin='left',            # Start from left side
    bsz=1,                   # Block size (usually 1)
    batch_inds=['s'],        # List of batch index names
    output_dims={'out'}      # Set of output dimension names (optional)
)
```

**Parameters:**
- `tn`: The tensor network to optimize
- `begin`: `'left'` or `'right'` - which side to start the sweep
- `bsz`: Block size (number of sites to optimize together, usually 1)
- `batch_inds`: List of index names that represent batch dimensions
- `output_dims`: Set of index names that represent output dimensions

---

## Environment Structure

At each position `i`, `env()` returns a tensor network with three components:

### 1. **_LEFT**: Contracted environment from sites 0 to i-1

```python
# At site i=2, _LEFT contains sites 0, 1 contracted together
env.move_to(2)
current_env = env()

left_tensors = [t for t in current_env.tensors if '_LEFT' in t.tags]
# left_tensors[0].inds includes:
#   - Bond connecting to site i (e.g., bond between site 1 and 2)
#   - Batch dimension 's'
#   - Output dimension 'out' (if present in sites 0..i-1)
```

### 2. **_RIGHT**: Contracted environment from sites i+1 to L-1

```python
right_tensors = [t for t in current_env.tensors if '_RIGHT' in t.tags]
# right_tensors[0].inds includes:
#   - Bond connecting to site i (e.g., bond between site 2 and 3)
#   - Batch dimension 's'
#   - Output dimension 'out' (if present in sites i+1..L-1)
```

### 3. **SAME_SITE**: Uncontracted tensors at site i

```python
site_tensors = [t for t in current_env.tensors 
                if f'I{i}' in t.tags 
                and '_LEFT' not in t.tags 
                and '_RIGHT' not in t.tags]
# Typically contains:
#   - MPS_i: The MPS tensor at site i
#   - INPUT_i: The input tensor at site i
```

### Visual Representation

```
Site i=2:

LEFT (sites 0,1)    SAME_SITE (site 2)           RIGHT (sites 3,4)
┌──────────────┐    ┌──────────────────┐         ┌──────────────┐
│              │    │  MPS_2           │         │              │
│  Sites 0,1   │────│  INPUT_2         │─────────│  Sites 3,4   │
│  contracted  │    │  (uncontracted)  │         │  contracted  │
│              │    │                  │         │              │
└──────────────┘    └──────────────────┘         └──────────────┘
     │                      │                          │
  bond(1→2)            all indices               bond(2→3)
  batch 's'            at site 2                 batch 's'
  output 'out'                                   output 'out'
  (if in 0,1)                                    (if in 3,4)
```

---

## Creating Holes

To optimize a specific tensor, you create a "hole" by removing the target from the environment.

### Basic Hole Creation

```python
# Move to target site
site_idx = 2
env.move_to(site_idx)

# Get environment
current_env = env()

# Create a copy and delete the target
env_with_hole = current_env.copy()
target_tag = f"MPS_{site_idx}"
env_with_hole.delete(target_tag)
```

### What the Hole Contains

After deletion, `env_with_hole` contains:
- **_LEFT**: Environment from sites 0 to i-1
- **_RIGHT**: Environment from sites i+1 to L-1  
- **Other tensors at site i**: e.g., `INPUT_i` (but NOT `MPS_i`)

### Outer Indices (Bonds to Target)

```python
# Get indices that connect to the deleted target
outer_inds = set(env_with_hole.outer_inds())

# These are the bonds that connected to MPS_i:
# - Left bond (from site i-1, if i > 0)
# - Right bond (to site i+1, if i < L-1)
# - Physical bond (to INPUT_i)
```

### Contracting the Environment

```python
# Collect all indices present in the environment
all_inds = set()
for t in env_with_hole.tensors:
    all_inds.update(t.inds)

# Prepare indices to keep
inds_to_keep = list(outer_inds)

# Add batch dimension if present
if 's' in all_inds and 's' not in inds_to_keep:
    inds_to_keep.append('s')

# Add output dimension if present
if 'out' in all_inds and 'out' not in inds_to_keep:
    inds_to_keep.append('out')

# Contract the environment
contracted_env = env_with_hole.contract(all, output_inds=inds_to_keep)

# contracted_env now has:
#   - Bonds to the target tensor
#   - Batch dimension 's' (shape: BATCH_SIZE)
#   - Output dimension 'out' (if present, shape: OUTPUT_DIM)
```

### Expected Shapes

**Example at site 2 (middle site with output):**

```python
# Before contraction:
env_with_hole.num_tensors  # 3 tensors: _LEFT, _RIGHT, INPUT_2

# After contraction:
contracted_env.inds  # ('bond_left', 'bond_right', 'phys_2', 's', 'out')
contracted_env.shape # (3, 3, 2, 100, 10)
#                      ↑  ↑  ↑  ↑    ↑
#                      │  │  │  │    └─ output dimension
#                      │  │  │  └────── batch dimension
#                      │  │  └───────── physical dimension
#                      │  └──────────── right bond
#                      └─────────────── left bond
```

**Key Point:** The output dimension 'out' is **only in the contracted environment** when the site with 'out' is NOT the current target. At the site with 'out', the output dimension is in the SAME_SITE tensor, not in LEFT or RIGHT.

---

## Movement Operations

### move_to(i)

Move directly to site `i`:

```python
env.move_to(2)  # Move to site 2
```

### move_right()

Move one site to the right:

```python
env.move_right()  # pos: i → i+1
```

### move_left()

Move one site to the left:

```python
env.move_left()  # pos: i → i-1
```

### Current Position

```python
current_pos = env.pos  # Get current position
```

---

## Complete Example

### Full Workflow: Optimizing an MPS with Batched Data

```python
import quimb.tensor as qt
import torch
from batch_moving_environment import BatchMovingEnvironment

# Setup
L = 4
BOND_DIM = 3
PHYS_DIM = 2
BATCH_SIZE = 100
OUTPUT_DIM = 10

# 1. Create MPS
psi = qt.MPS_rand_state(L, bond_dim=BOND_DIM, phys_dim=PHYS_DIM)
psi.apply_to_arrays(lambda x: torch.tensor(x, dtype=torch.float32))
psi.reindex({f"k{i}": f"phys_{i}" for i in range(L)}, inplace=True)

# Add unique tags
for i in range(L):
    psi.add_tag(f"MPS_{i}", where=f"I{i}")

# 2. Add output dimension to middle site (BEFORE torch conversion if needed)
middle_site = 2
middle_tensor = psi[f'I{middle_site}']
middle_tensor.new_ind('out', size=OUTPUT_DIM, axis=-1)

# 3. Create batch inputs
inputs = []
for i in range(L):
    inp = qt.Tensor(
        data=torch.rand(BATCH_SIZE, PHYS_DIM, dtype=torch.float32),
        inds=['s', f'phys_{i}'],
        tags={f'I{i}', 'INPUT', f'INPUT_{i}'}
    )
    inputs.append(inp)

# 4. Build tensor network
tn = psi.copy()
for inp in inputs:
    tn.add_tensor(inp)

# 5. Initialize environment
env = BatchMovingEnvironment(
    tn,
    begin='left',
    bsz=1,
    batch_inds=['s'],
    output_dims={'out'}
)

# 6. Left-to-right sweep
for site_idx in range(L):
    print(f"Optimizing site {site_idx}")
    
    # Move to site
    env.move_to(site_idx)
    
    # Get environment
    current_env = env()
    
    # Create hole
    env_with_hole = current_env.copy()
    target_tag = f"MPS_{site_idx}"
    env_with_hole.delete(target_tag)
    
    # Get target tensor
    target = psi[f"I{site_idx}"]
    
    # Contract environment
    outer_inds = list(env_with_hole.outer_inds())
    all_inds = set().union(*(t.inds for t in env_with_hole.tensors))
    
    if 's' in all_inds and 's' not in outer_inds:
        outer_inds.append('s')
    if 'out' in all_inds and 'out' not in outer_inds:
        outer_inds.append('out')
    
    contracted_env = env_with_hole.contract(all, output_inds=outer_inds)
    
    # Compute gradient/update (example)
    # gradient = contracted_env & some_derivative_tensor
    # new_target = compute_update(gradient, target)
    # target.modify(data=new_target.data)
    
    print(f"  Environment shape: {contracted_env.shape}")
    print(f"  Environment indices: {contracted_env.inds}")

# 7. Right-to-left sweep
for site_idx in reversed(range(L)):
    print(f"Optimizing site {site_idx} (backward)")
    
    # Same process as above
    env.move_to(site_idx)
    # ... optimization code ...
```

---

## Advanced Usage

### Working with Multiple Batch Dimensions

```python
env = BatchMovingEnvironment(
    tn,
    begin='left',
    bsz=1,
    batch_inds=['s', 'batch2'],  # Multiple batch dimensions
    output_dims={'out'}
)
```

### Working with Multiple Output Dimensions

```python
# Add multiple output dimensions
tensor.new_ind('out1', size=10)
tensor.new_ind('out2', size=5)

env = BatchMovingEnvironment(
    tn,
    begin='left',
    bsz=1,
    batch_inds=['s'],
    output_dims={'out1', 'out2'}  # Multiple outputs
)
```

### CMPO2 Structure (Two MPS Networks)

For structures with two MPS networks (e.g., pixel MPS and patch MPS):

```python
# Create two MPS
psi = qt.MPS_rand_state(L, bond_dim=3, phys_dim=4)  # Pixel MPS
phi = qt.MPS_rand_state(L, bond_dim=3, phys_dim=5)  # Patch MPS

# Reindex to avoid conflicts
psi.reindex({f"k{i}": f"{i}_pixels" for i in range(L)}, inplace=True)
phi.reindex({f"k{i}": f"{i}_patches" for i in range(L)}, inplace=True)

# Add unique tags
for i in range(L):
    psi.add_tag(f"{i}_Pi", where=f"I{i}")
    phi.add_tag(f"{i}_Pa", where=f"I{i}")

# Combine
tn = psi & phi

# Create inputs that connect to BOTH MPS
for i in range(L):
    inp = qt.Tensor(
        data=torch.rand(BATCH_SIZE, 5, 4, dtype=torch.float32),
        inds=['s', f'{i}_patches', f'{i}_pixels'],  # Connects to both!
        tags={f'I{i}', 'INPUT', f'INPUT_{i}'}
    )
    tn.add_tensor(inp)

# Initialize environment (same as before)
env = BatchMovingEnvironment(tn, begin='left', bsz=1, batch_inds=['s'])
```

---

## Troubleshooting

### Problem: "Index appears more than twice"

**Error:**
```
ValueError: The index s appears more than twice!
```

**Cause:** You're trying to contract a hyper-edge network without specifying `output_inds`.

**Solution:** Always use `contract(all, output_inds=...)`:
```python
# WRONG
contracted = env_with_hole.contract(output_inds=inds_to_keep)

# CORRECT
contracted = env_with_hole.contract(all, output_inds=inds_to_keep)
```

### Problem: Missing bonds in outer_inds

**Symptom:** After deleting the target, `outer_inds()` doesn't include expected MPS bonds.

**Cause:** This was a bug in the original `_get_inds_to_keep` implementation (now fixed).

**Solution:** Make sure you're using the **fixed version** of `BatchMovingEnvironment` that includes the `site_tag` parameter in `_get_inds_to_keep()`.

### Problem: Batch dimension disappears

**Symptom:** After contracting, the batch dimension 's' is missing.

**Cause:** Forgot to add batch dimension to `inds_to_keep`.

**Solution:**
```python
outer_inds = list(env_with_hole.outer_inds())
all_inds = set().union(*(t.inds for t in env_with_hole.tensors))

# CRITICAL: Add batch dimension
if 's' in all_inds and 's' not in outer_inds:
    outer_inds.append('s')

contracted = env_with_hole.contract(all, output_inds=outer_inds)
```

### Problem: Output dimension in wrong place

**Symptom:** The output dimension appears where it shouldn't, or doesn't appear where it should.

**Expected Behavior:**
- At the site WITH output: output dimension is in SAME_SITE, NOT in LEFT/RIGHT
- At sites BEFORE the output site: output dimension is in RIGHT
- At sites AFTER the output site: output dimension is in LEFT

**Check:** Look at individual _LEFT, _RIGHT tensors:
```python
left_tensors = [t for t in current_env.tensors if '_LEFT' in t.tags]
right_tensors = [t for t in current_env.tensors if '_RIGHT' in t.tags]

if left_tensors:
    print(f"LEFT indices: {left_tensors[0].inds}")
if right_tensors:
    print(f"RIGHT indices: {right_tensors[0].inds}")
```

---

## Best Practices

1. **Always add unique tags** to each site (e.g., `MPS_0`, `MPS_1`, etc.)
2. **Use descriptive index names** (e.g., `phys_0` instead of `k0`)
3. **Always specify `batch_inds`** even if empty: `batch_inds=[]`
4. **Use `contract(all, output_inds=...)` ** for hyper-edge networks
5. **Check indices before contracting**:
   ```python
   print(f"Outer: {env_with_hole.outer_inds()}")
   print(f"Will keep: {inds_to_keep}")
   ```
6. **Verify shapes after contraction**:
   ```python
   assert contracted.shape[contracted.inds.index('s')] == BATCH_SIZE
   ```

---

## Summary Table

| Component | What It Contains | When to Use |
|-----------|-----------------|-------------|
| **_LEFT** | Sites 0 to i-1, contracted | Automatic - part of `env()` |
| **_RIGHT** | Sites i+1 to L-1, contracted | Automatic - part of `env()` |
| **SAME_SITE** | Uncontracted tensors at site i | Automatic - part of `env()` |
| **Hole** | LEFT + RIGHT + other site tensors (target deleted) | For optimization |
| **outer_inds()** | Bonds connecting to deleted target | Always use after creating hole |
| **batch_inds** | Dimensions to preserve (e.g., 's') | Pass to `__init__` |
| **output_dims** | Output dimensions to preserve (e.g., 'out') | Pass to `__init__` |

---

## References

- `model/batch_moving_environment.py` - Implementation
- `test_left_right_site_bonds.py` - Test showing correct bond preservation
- `test_output_dim_bonds.py` - Test with output dimensions
- `verify_hole_calculation.py` - Test of hole creation at each site

---

**Version:** 1.0  
**Last Updated:** December 2024  
**Status:** ✅ Verified and Working
