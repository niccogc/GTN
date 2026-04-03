# CPD-Asymmetric Model Implementation Plan

## Goal

Implement a **Canonical Polyadic Decomposition (CPD) Asymmetric** tensor network model for the GTN framework. Unlike MPS/MPO models that use bond indices connecting neighboring tensors, CPDA uses a shared **rank index** across all factor matrices, with the final output computed via elementwise multiplication over the rank dimension.

## CPD-Asymmetric Structure

### Tensor Network Diagram
```
     x0    x1    x2    ...   xL-1
     |     |     |           |
  [Node0][Node1][Node2]...[NodeL-1]
     \     |     |           /
      \    |    /           /
       \   |   /           /
        \  |  /           /
         [r]  (shared rank index)
          |
        [out] (output at L-1 node)
```

### Node Structure
- **Node_i**: shape `(phys_dim, rank)` with indices `(x{i}, r)`
- All nodes share the same rank index `r`
- Output dimension added to one node (last node): `(phys_dim, rank, out)`

### Contraction
- Full contraction: `tn.contract()` contracts over `r`, producing output
- For NTN environment: elementwise multiplication of all other nodes **without** summing over rank
  - Environment for Node_i = product of all Node_j (j != i) over physical indices only
  - The rank index `r` remains open (not contracted)

## Implementation Steps

### 1. Create `model/standard/CPD.py`

```python
class CPDA:
    def __init__(
        self,
        L: int,                    # Number of sites/features
        rank: int,                 # CPD rank (analogous to bond_dim)
        phys_dim: int,             # Physical dimension per site
        output_dim: int,           # Output dimension (classes or 1 for regression)
        output_site: int = None,   # Which node gets output (default: last)
        init_strength: float = 0.001,
        use_tn_normalization: bool = True,
        ...
    ):
        # Create L tensors, each with shape (phys_dim, rank)
        # All share index "r" for rank
        # One node also has "out" index
        
        self.tn = qt.TensorNetwork(tensors)
        self.input_labels = [f"x{i}" for i in range(L)]
        self.input_dims = [f"x{i}" for i in range(L)]
        self.output_dims = ["out"]
```
# Most Important

Verify if NTN already handles properly the new CPDA class

Otherwise We need a class CPDA_NTN that inherits from NTN and modify the related forward and batch environment functions!!

Otherwise for GTN we need same idea, a class CPDA_GTN inheriting from GTN but with a different forward.

First and foremost see that, the existing NTN and GTN, are already handling the CPD structure as expect for the forward and computing environment. Then we can treat it as a standard other model

### 2. Key Design Decisions

| Aspect | Decision |
|--------|----------|
| Rank index name | `"r"` (shared across all nodes) |
| Physical indices | `"x0"`, `"x1"`, ..., `"xL-1"` |
| Output index | `"out"` on output_site node |
| Node tags | `"Node0"`, `"Node1"`, ..., `"NodeL-1"` |
| Contraction | `tn.contract()` - quimb handles shared `r` correctly |

### 3. Environment Computation for NTN

The NTN base class computes environments by:
1. Creating a copy of the TN with inputs
2. Removing the target node
3. Contracting the rest

For CPD, when computing the environment for Node_i:
- All other nodes share index `r`
- The environment should be the **elementwise product** over `r` (not summed)
- Result shape: `(batch, phys_dim_i, rank, [out])`

This should work automatically with quimb's contraction since:
- Removing Node_i leaves nodes with indices `(x_j, r)` for j != i
- Plus input tensors with indices `(batch, x_j)`
- Contracting gives tensor with remaining open indices including `r`

### 4. Update Model Registry

**`model/standard/__init__.py`**:
```python
from model.standard.CPD import CPDA
```

**`run.py`**:
```python
```
To add CPDA model

### 5. Create Config File

**`conf/model/cpda.yaml`**:
```yaml
defaults:
  - _base

name: CPDA
bond_dim: 4       # Alias for rank (for compatibility)
L: ${dataset.n_features}
output_site: null # Default to last site
```

## Testing Strategy

