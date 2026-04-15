# GTN Model Architecture Analysis

## Executive Summary

The GTN (Gradient Tensor Network) framework implements tensor network models for machine learning with specialized training via alternating least squares (ALS) and Newton-based optimization. The architecture consists of:

1. **MPO2 Models** - Tensor network construction with proper indexing
2. **NTN (Newton Tensor Network)** - Training engine with environment-based optimization
3. **Input Preparation** - Data handling with bias term support
4. **Initialization** - Multiple normalization strategies

---

## 1. MPO2 Model Architecture (model/standard/MPO2_models.py)

### 1.1 MPO2 Class - Simple MPS with Output Dimension

**Purpose**: Standard Matrix Product State (MPS) chain with one site containing the output dimension.

#### Tensor Construction

**For L=1 (Single Site)**:
```
Shape: (phys_dim, output_dim)
Indices: ("x0", "out")
Tags: {"Node0"}
Data: torch.randn(phys_dim, output_dim) * base_init
```

**For L>1 (Chain)**:

| Site Position | Shape | Indices | Tags |
|---|---|---|---|
| i=0 (left) | (phys_dim, bond_dim) | ("x0", "b0") | {"Node0"} |
| 0 < i < L-1 (middle) | (bond_dim, phys_dim, bond_dim) | ("b{i-1}", "x{i}", "b{i}") | {"Node{i}"} |
| i=L-1 (right) | (bond_dim, phys_dim) | ("b{L-2}", "x{L-1}") | {"Node{L-1}"} |

**Output Dimension Addition**:
- If `i == output_site`: append `output_dim` to shape and "out" to indices
- Example at output_site=L-1: shape becomes (bond_dim, phys_dim, output_dim)
- Indices become: ("b{L-2}", "x{L-1}", "out")

#### Index Naming Convention

| Index Type | Pattern | Meaning |
|---|---|---|
| Physical | `x{i}` | Input feature at site i |
| Bond | `b{i}` | Bond connecting sites i and i+1 |
| Output | `out` | Output dimension (single, shared across network) |
| Tag | `Node{i}` | Unique identifier for site i |

#### Initialization Parameters

```python
base_init = 0.1 if use_tn_normalization else init_strength
# Default: init_strength=0.001 (only used if use_tn_normalization=False)
# With normalization: base_init=0.1 (before scaling)
```

#### Normalization Strategies

**Option 1: Output-Based Normalization** (if sample_inputs provided):
```python
normalize_tn_output(
    tn,
    sample_inputs,
    output_dims=["out"],
    batch_dim="s",
    target_std=0.1  # default
)
```
- Computes predictions on sample inputs
- Scales all trainable tensors: scale_factor = target_std / current_std
- Eliminates seed-dependent collapses

**Option 2: Frobenius Norm Normalization** (if no sample_inputs):
```python
target_norm = np.sqrt(L * bond_dim * phys_dim)
normalize_tn_frobenius(tn, target_norm=target_norm)
```
- Scales to match Frobenius norm of √(L·bond_dim·phys_dim)
- Deterministic, doesn't require data

#### Example: L=3, bond_dim=4, phys_dim=2, output_dim=3

```
Node0: (2, 4)     indices: ("x0", "b0")
Node1: (4, 2, 4)  indices: ("b0", "x1", "b1")
Node2: (4, 2, 3)  indices: ("b1", "x2", "out")  <- output_site=2
```

---

### 1.2 CMPO2 Class - Cross MPS (Pixels and Patches)

**Purpose**: Two MPS layers that cross-connect for multi-scale processing.

#### Structure

```
Pixel MPS (psi):  L sites, phys_dim=phys_dim_pixels
Patch MPS (phi):  L sites, phys_dim=phys_dim_patches
```

#### Index Naming

| Component | Index Pattern | Example |
|---|---|---|
| Pixel MPS | `{i}_pixels` | "0_pixels", "1_pixels" |
| Patch MPS | `{i}_patches` | "0_patches", "1_patches" |
| Pixel Tags | `{i}_Pi` | "0_Pi", "1_Pi" |
| Patch Tags | `{i}_Pa` | "0_Pa", "1_Pa" |
| Output | `out` | Added to pixel MPS at output_site |

#### Input Labels (for builder)

```python
input_labels = [[0, (f"{i}_patches", f"{i}_pixels")] for i in range(L)]
# Explicit format: [source_idx, (indices...)]
# source_idx=0: use first input tensor
# indices: tuple of dimension names to connect
```

---

### 1.3 LMPO2 Class - Linear MPO + MPS

**Purpose**: Dimensionality reduction via MPO, then output via MPS.

#### Two-Layer Structure

**Layer 1: MPO (Dimensionality Reduction)**
```
Input: phys_dim
Output: reduced_dim
Reduction factor: reduced_dim / phys_dim
```

| Site | Shape | Indices | Tags |
|---|---|---|---|
| i=0 | (phys_dim, reduced_dim, bond_dim_mpo) | ("{i}_in", "{i}_reduced", "b_mpo_{i}") | {"{i}_MPO"} |
| 0<i<L-1 | (bond_dim_mpo, phys_dim, reduced_dim, bond_dim_mpo) | ("b_mpo_{i-1}", "{i}_in", "{i}_reduced", "b_mpo_{i}") | {"{i}_MPO"} |
| i=L-1 | (bond_dim_mpo, phys_dim, reduced_dim) | ("b_mpo_{i-1}", "{i}_in", "{i}_reduced") | {"{i}_MPO"} |

**Layer 2: MPS (Output)**
```
Input: reduced_dim
Output: output_dim
```

| Site | Shape | Indices | Tags |
|---|---|---|---|
| i=0 | (reduced_dim, bond_dim) | ("{i}_reduced", "b_mps_{i}") | {"{i}_MPS"} |
| 0<i<L-1 | (bond_dim, reduced_dim, bond_dim) | ("b_mps_{i-1}", "{i}_reduced", "b_mps_{i}") | {"{i}_MPS"} |
| i=L-1 | (bond_dim, reduced_dim, output_dim) | ("b_mps_{i-1}", "{i}_reduced", "out") | {"{i}_MPS"} |

#### Input Labels

```python
input_labels = [f"{i}_in" for i in range(L)]
```

---

### 1.4 MMPO2 Class - Masking MPO + MPS

**Purpose**: Non-trainable cumulative sum mask + trainable MPS.

#### Mask Structure

**Heaviside Matrix H**:
```python
H[i, j] = 1.0 if j >= i else 0.0
# Upper triangular matrix (causal mask)
```

**Kronecker Delta Tensor Δ**:
```python
Δ[k, k, k] = 1.0  (for appropriate dimensions)
# Diagonal tensor
```

**Mask Tensor C**:
```
C^{i_in, i_out}_{b_left, b_right} = sum_k H_{b_left, k} * Δ_{k, i_in, i_out, b_right}
```

#### Mask Tensor Shapes

| Site | Shape | Indices | Tags |
|---|---|---|---|
| i=0 | (phys_dim, phys_dim, phys_dim) | ("{i}_in", "{i}_masked", "b_mask_{i}") | {"{i}_Mask", "NT"} |
| 0<i<L-1 | (phys_dim, phys_dim, phys_dim, phys_dim) | ("b_mask_{i-1}", "{i}_in", "{i}_masked", "b_mask_{i}") | {"{i}_Mask", "NT"} |
| i=L-1 | (phys_dim, phys_dim, phys_dim) | ("b_mask_{i-1}", "{i}_in", "{i}_masked") | {"{i}_Mask", "NT"} |

**Key Feature**: "NT" tag marks non-trainable tensors (excluded from optimization).

#### MPS Layer (Trainable)

Same structure as MPO2, but indices connect to masked outputs:

| Site | Shape | Indices | Tags |
|---|---|---|---|
| i=0 | (phys_dim, bond_dim) | ("{i}_masked", "b_mps_{i}") | {"{i}_MPS"} |
| 0<i<L-1 | (bond_dim, phys_dim, bond_dim) | ("b_mps_{i-1}", "{i}_masked", "b_mps_{i}") | {"{i}_MPS"} |
| i=L-1 | (bond_dim, phys_dim, output_dim) | ("b_mps_{i-1}", "{i}_masked", "out") | {"{i}_MPS"} |

---

## 2. NTN Training Engine (model/base/NTN.py)

### 2.1 Forward Pass

#### Basic Forward Pass

```python
def _batch_forward(inputs: List[qt.Tensor], tn, output_inds: List[str]) -> qt.Tensor:
    full_tn = tn & inputs  # Contract input tensors into network
    res = full_tn.contract(output_inds=output_inds)
    if len(output_inds) > 0:
        res.transpose_(*output_inds)
    return res
```

**Output Indices Logic**:
```python
if sum_over_output:
    if sum_over_batch:
        target_inds = []  # Scalar output
    else:
        target_inds = [batch_dim]  # (batch,)
elif sum_over_batch:
    target_inds = output_dimensions  # (out_dim,)
else:
    target_inds = [batch_dim] + output_dimensions  # (batch, out_dim)
```

#### Environment-Based Forward Pass

```python
def forward_from_environment(env, node_tag, node_tensor, sum_over_batch=False):
    """
    Fast forward pass using pre-computed environment.
    
    Key advantage: Once env is computed, can evaluate with different node values
    without re-contracting entire network.
    """
    if sum_over_batch:
        output_inds = output_dimensions
    else:
        output_inds = [batch_dim] + output_dimensions
    
    y_pred = (env & node_tensor).contract(output_inds=output_inds)
    if len(output_inds) > 0:
        y_pred.transpose_(*output_inds)
    return y_pred
```

### 2.2 Environment Computation

#### _batch_environment Method

```python
def _batch_environment(inputs, tn, target_tag, sum_over_batch=False, sum_over_output=False):
    """
    Computes environment tensor: all network except target node.
    
    Environment indices formula:
    env_inds = {batch_dim} ∪ node_inds ∪ out_labels - (node_inds ∩ out_labels)
    
    This unified formula works for both:
    - MPS-style networks (bond indices)
    - CPDA-style networks (shared rank index)
    """
    target_tensor = tn[target_tag]
    node_inds = set(target_tensor.inds)
    out_labels = set(output_dimensions)
    
    env_tn = tn & inputs
    env_tn.delete(target_tag)
    
    # Compute output indices
    intersection = node_inds & out_labels
    env_inds = ({batch_dim} | node_inds | out_labels) - intersection
    
    if sum_over_batch and batch_dim in env_inds:
        env_inds.remove(batch_dim)
    
    if sum_over_output:
        for out_dim in output_dimensions:
            if out_dim in env_inds:
                env_inds.remove(out_dim)
    
    env_tensor = env_tn.contract(output_inds=env_inds)
    return env_tensor
```

**Example**: For MPO2 with output_site=L-1:
- target_tensor indices: ("b{L-2}", "x{L-1}", "out")
- node_inds: {"b{L-2}", "x{L-1}", "out"}
- out_labels: {"out"}
- intersection: {"out"}
- env_inds: {"s", "b{L-2}", "x{L-1}"} - {"out"} = {"s", "b{L-2}", "x{L-1}"}

### 2.3 Gradient and Hessian Computation

#### _batch_node_derivatives Method

```python
def _batch_node_derivatives(inputs, y_true, node_tag):
    """
    Computes (Node_Grad, Node_Hess) for a single batch.
    
    Process:
    1. Compute environment E (batch, out, node_bonds)
    2. Forward pass via E -> y_pred
    3. Loss derivatives -> dL/dy, d²L/dy²
    4. Node Jacobian = E * dL/dy
    5. Node Hessian = (E * d²L/dy²) * E^T
    """
    # Step 1: Environment
    env = _batch_environment(inputs, tn, target_tag=node_tag, 
                            sum_over_batch=False, sum_over_output=False)
    target_tensor = tn[node_tag]
    
    # Step 2: Forward pass
    y_pred = forward_from_environment(env, node_tag, target_tensor, 
                                     sum_over_batch=False)
    
    # Step 3: Loss derivatives
    dL_dy, d2L_dy2 = loss.get_derivatives(
        y_pred, y_true,
        backend=backend,
        batch_dim=batch_dim,
        output_dims=output_dimensions,
        return_hessian_diagonal=False,
        total_samples=train_data.samples
    )
    
    # Step 4: Node Jacobian
    grad_tn = env & dL_dy
    node_inds = target_tensor.inds
    node_grad = grad_tn.contract(output_inds=node_inds)
    
    # Step 5: Node Hessian
    out_row_inds = output_dimensions
    out_col_inds = [x + "_prime" for x in output_dimensions]
    
    d2L_tensor = qt.Tensor(
        d2L_dy2.data, 
        inds=[batch_dim] + out_row_inds + out_col_inds
    )
    env_right = _prime_indices_tensor(env, exclude_indices=[batch_dim])
    
    hess_tn = env & d2L_tensor & env_right
    hess_out_inds = list(node_inds) + [f"{x}_prime" for x in node_inds]
    node_hess = hess_tn.contract(output_inds=hess_out_inds)
    
    return node_grad, node_hess
```

#### Tensor Shapes in Gradient/Hessian Computation

**Environment Tensor**:
```
Shape: (batch, *node_inds, *out_labels)
Example for MPO2: (batch, b{L-2}, x{L-1}, out)
```

**Loss Derivatives**:
```
dL_dy shape: (batch, *output_dims)
d2L_dy2 shape: (batch, *output_dims, *output_dims_prime)
```

**Node Gradient**:
```
Shape: (*node_inds)
Example: (b{L-2}, x{L-1}, out)
```

**Node Hessian**:
```
Shape: (*node_inds, *node_inds_prime)
Example: (b{L-2}, x{L-1}, out, b{L-2}_prime, x{L-1}_prime, out_prime)
```

### 2.4 Node Update Computation

#### _get_node_update Method

```python
def _get_node_update(node_tag, regularize=True, jitter=1e-6):
    """
    Computes Newton update: Δ = -H^{-1} * b
    where H is Hessian, b is gradient
    """
    b, H = _compute_H_b(node_tag)
    
    # Fuse indices for linear algebra
    variational_ind = b.inds
    map_H = {
        "rows": variational_ind,
        "cols": [i + "_prime" for i in variational_ind]
    }
    map_b = {"cols": variational_ind}
    
    H.fuse(map_H, inplace=True)
    b.fuse(map_b, inplace=True)
    
    # Convert to dense matrices
    matrix_data = H.to_dense(["rows"], ["cols"])  # (N, N)
    gradient_vector = b.to_dense(["cols"])  # (N,)
    
    # Regularization: H += jitter * I, b += jitter * w_old
    if regularize:
        current_node = tn[node_tag].copy()
        current_node.fuse(map_b, inplace=True)
        old_weight = current_node.to_dense(["cols"])
        
        scaled_jitter = 2 * jitter
        matrix_data.diagonal().add_(scaled_jitter)
        gradient_vector = gradient_vector + scaled_jitter * old_weight
    
    # Solve: H * x = -b
    tensor_node_data = solve_linear_system(matrix_data, -gradient_vector)
    
    # Unfuse back to tensor shape
    update_node = qt.Tensor(tensor_node_data, inds=["cols"], tags=tn[node_tag].tags)
    update_node.unfuse({"cols": variational_ind}, shape_map=shape_map, inplace=True)
    
    return update_node
```

#### Regularization Strategy

```python
# L2 regularization: minimize ||w||² + jitter * ||w - w_old||²
# Equivalent to: (H + 2*jitter*I) * w = -b + 2*jitter*w_old

scaled_jitter = 2 * jitter
H_reg = H + scaled_jitter * I
b_reg = -b + scaled_jitter * w_old
```

### 2.5 Training Loop (fit method)

#### Algorithm Selection

```python
from model.losses import MSELoss, MAELoss, HuberLoss

use_lstsq = isinstance(loss, (MSELoss, MAELoss, HuberLoss))

if use_lstsq:
    # Regression: Direct least squares solver
    update_tn_node_optimum(node_tag, regularize, jitter)
else:
    # Classification: Newton-based optimization
    update_tn_node(node_tag, regularize, jitter, adaptive_jitter)
```

#### Sweep Order

```python
trainable_nodes = _get_trainable_nodes()
back_sweep = trainable_nodes[-2:0:-1]  # Reverse, excluding first and last
full_sweep_order = trainable_nodes + back_sweep
```

Example for L=3: [Node0, Node1, Node2, Node1, Node0]

#### Early Stopping

```python
best_val_quality = compute_quality(scores_val)
patience_counter = 0

for epoch in range(n_epochs):
    # ... training ...
    current_val_quality = compute_quality(scores_val)
    
    if current_val_quality > best_val_quality + min_delta:
        best_val_quality = current_val_quality
        patience_counter = 0
    else:
        patience_counter += 1
    
    if patience is not None and patience_counter >= patience:
        return best_scores_train, best_scores_val
```

---

## 3. Input Preparation (model/utils.py)

### 3.1 create_inputs Function

```python
def create_inputs(
    X,                          # (samples, features)
    y,                          # (samples,) or (samples, output_dim)
    input_labels=None,
    output_labels=None,
    batch_size=32,
    batch_dim="s",
    append_bias=True,           # KEY PARAMETER
    encoding=None,              # "polynomial" or "fourier"
    poly_degree=None
) -> Inputs:
```

#### Bias Term Handling

**With append_bias=True** (default):
```python
if append_bias:
    n_samples = X.shape[0]
    X = torch.cat([X, torch.ones(n_samples, 1, dtype=X.dtype, device=X.device)], dim=1)
    # X shape: (samples, features+1)
    # Last column is all ones (bias term)
```

**Input Tensor Structure**:
```python
Inputs(
    inputs=[X],  # Single input tensor with bias appended
    outputs=[y],
    outputs_labels=output_labels,
    input_labels=input_labels,
    batch_dim=batch_dim,
    batch_size=batch_size
)
```

#### Polynomial Encoding

```python
def encode_polynomial(X: torch.Tensor, degree: int) -> torch.Tensor:
    """
    Transform X (samples, features) -> (samples, features, degree+1)
    
    For each feature x_i: [1, x_i, x_i^2, ..., x_i^degree]
    """
    n_samples, n_features = X.shape
    powers = torch.arange(degree + 1, dtype=X.dtype, device=X.device)
    X_expanded = X.unsqueeze(-1)  # (samples, features, 1)
    return X_expanded ** powers   # (samples, features, degree+1)
```

**Example**: X shape (100, 3), degree=2
```
Output shape: (100, 3, 3)
For feature i: [x_i^0, x_i^1, x_i^2] = [1, x_i, x_i^2]
```

#### Fourier Encoding

```python
def encode_fourier(X: torch.Tensor) -> torch.Tensor:
    """
    Transform X (samples, features) -> (samples, features, 2)
    
    For each feature x_i: [cos(x_i * π/2), sin(x_i * π/2)]
    """
    scaled = X * (math.pi / 2)
    cos_features = torch.cos(scaled)
    sin_features = torch.sin(scaled)
    return torch.stack([cos_features, sin_features], dim=-1)
```

### 3.2 Inputs Class (model/builder.py)

#### Batch Creation

```python
class Inputs:
    def __init__(self, inputs, outputs, outputs_labels, input_labels, 
                 batch_dim="s", batch_size=None):
        self.inputs_data = inputs
        self.outputs_data = outputs
        self.outputs_labels = outputs_labels
        self.input_labels = input_labels
        self.batch_dim = batch_dim
        
        self.batch_size = inputs[0].shape[0] if batch_size is None else batch_size
        self.samples = outputs[0].shape[0]
        
        self.batches = self._create_batches()
```

#### Batch Preparation

```python
def _prepare_batch(self, input_data: Dict[int, Any]) -> List[qt.Tensor]:
    """
    Constructs QT tensors based on input_labels definitions.
    """
    tensors = []
    
    for i, definition in enumerate(self.input_labels):
        # Parse definition (explicit or implicit)
        if isinstance(definition, list):
            source_idx = definition[0]
            inds_def = definition[1]
        else:
            source_idx = 0 if len(self.inputs_data) == 1 else i
            inds_def = definition
        
        # Parse indices
        if isinstance(inds_def, str):
            inds = (self.batch_dim, inds_def)
            tag_suffix = inds_def
        else:
            inds = (self.batch_dim, *inds_def)
            tag_suffix = "_".join(inds_def)
        
        # Fetch data
        data = input_data[source_idx]
        
        # Create tensor
        tags = {f'input_{tag_suffix}', f'I{i}'}
        tensor = qt.Tensor(data=data, inds=inds, tags=tags)
        tensors.append(tensor)
    
    return tensors
```

#### Tensor Structure Example

**For single input with bias**:
```
Input tensor:
  Shape: (batch_size, features+1)
  Indices: ("s", "x")  # batch_dim="s", input_label="x"
  Tags: {"input_x", "I0"}
```

**For CMPO2 with two inputs**:
```
Input tensor 0:
  Shape: (batch_size, phys_dim_patches)
  Indices: ("s", "0_patches")
  Tags: {"input_0_patches", "I0"}

Input tensor 1:
  Shape: (batch_size, phys_dim_pixels)
  Indices: ("s", "0_pixels")
  Tags: {"input_0_pixels", "I1"}
```

---

## 4. Initialization Strategies (model/initialization.py)

### 4.1 normalize_tn_output

```python
def normalize_tn_output(tn, input_samples, output_dims, batch_dim="s", 
                       target_std=0.1, max_samples=1000, inplace=True):
    """
    Normalize TN so initial outputs have target_std on sample inputs.
    
    Strategy:
    1. Compute predictions on sample inputs
    2. Measure output std
    3. Scale all trainable tensors: scale_factor = target_std / current_std
    """
    # Forward pass on samples
    full_tn = tn & inputs
    y_pred = full_tn.contract(output_inds=[batch_dim] + output_dims)
    
    # Measure current std
    current_std = y_pred.data.std().item()
    
    # Compute scale factor
    if current_std < 1e-10:
        scale_factor = target_std / 0.01
    else:
        scale_factor = target_std / current_std
    
    # Apply scaling (skip NT-tagged tensors)
    for tensor in tn:
        if "NT" not in tensor.tags:
            tensor.modify(data=tensor.data * scale_factor)
    
    return scale_factor
```

### 4.2 normalize_tn_frobenius

```python
def normalize_tn_frobenius(tn, target_norm=1.0, exclude_tags=None, inplace=True):
    """
    Normalize TN by Frobenius norm.
    
    Scales all trainable tensors so total Frobenius norm equals target_norm.
    """
    if exclude_tags is None:
        exclude_tags = ["NT"]
    
    # Compute total Frobenius norm
    total_norm_sq = 0.0
    for tensor in tn:
        if not any(tag in tensor.tags for tag in exclude_tags):
            total_norm_sq += (tensor.data**2).sum().item()
    
    current_norm = np.sqrt(total_norm_sq)
    
    # Compute scale factor
    if current_norm < 1e-10:
        return 1.0
    
    scale_factor = target_norm / current_norm
    
    # Apply scaling
    for tensor in tn:
        if not any(tag in tensor.tags for tag in exclude_tags):
            tensor.modify(data=tensor.data * scale_factor)
    
    return scale_factor
```

### 4.3 Initialization in MPO2

```python
# In MPO2.__init__:
base_init = 0.1 if use_tn_normalization else init_strength

# Create tensors with base_init
data = torch.randn(*shape) * base_init

# Then normalize
if use_tn_normalization:
    if sample_inputs is not None:
        normalize_tn_output(tn, sample_inputs, output_dims=["out"], 
                          batch_dim="s", target_std=tn_target_std)
    else:
        target_norm = np.sqrt(L * bond_dim * phys_dim)
        normalize_tn_frobenius(tn, target_norm=target_norm)
```

---

## 5. Loss Functions and Derivatives (model/losses.py)

### 5.1 MSELoss

```python
class MSELoss(nn.MSELoss, TNLoss):
    use_diagonal_hessian = True
    
    def get_derivatives(y_pred, y_true, backend='numpy', batch_dim='batch', 
                       output_dims=None, return_hessian_diagonal=None, 
                       total_samples=None):
        """
        For MSE: L = (1/N) * sum((y_pred - y_true)²)
        
        Gradient: dL/dy_pred = 2 * (y_pred - y_true)
        Hessian (diagonal): d²L/dy_pred² = 2 (constant)
        """
        y_pred_th = _to_torch(y_pred, requires_grad=False)
        y_true_th = _to_torch(y_true, requires_grad=False)
        
        # Gradient
        grad_th = (y_pred_th - y_true_th) * 2
        
        # Hessian (diagonal)
        if return_hessian_diagonal:
            hess_th = torch.ones_like(y_pred_th) * 2
        else:
            # Full Hessian: diagonal matrix with 2s
            batch_sz, out_size = y_pred_th.shape[0], y_pred_th.shape[1]
            hess_th = torch.zeros(batch_sz, out_size, out_size)
            for i in range(out_size):
                hess_th[:, i, i] = 2.0
        
        return qt.Tensor(grad_data, inds=grad_inds), qt.Tensor(hess_data, inds=hess_inds)
```

### 5.2 CrossEntropyLoss

```python
class CrossEntropyLoss(nn.CrossEntropyLoss, TNLoss):
    use_diagonal_hessian = False  # Full Hessian required!
    
    def get_derivatives(y_pred, y_true, backend='numpy', batch_dim='batch',
                       output_dims=None, return_hessian_diagonal=None,
                       total_samples=None):
        """
        For cross-entropy: L = -sum(y_true * log(softmax(y_pred)))
        
        Gradient: dL/dz_i = p_i - y_i  (where p = softmax(logits))
        Hessian: d²L/dz_i dz_j = p_i * (δ_ij - p_j)  (FULL MATRIX!)
        
        Note: Softmax coupling makes Hessian inherently full (not diagonal).
        """
        y_pred_th = _to_torch(y_pred, requires_grad=True)
        y_true_th = _to_torch(y_true, requires_grad=False)
        
        # Compute loss
        loss_val = nn.CrossEntropyLoss.__call__(self, y_pred_th, y_true_indices)
        
        # Gradient via autograd
        grad_th = torch.autograd.grad(loss_val, y_pred_th, create_graph=True)[0]
        
        # Hessian via autograd (full matrix)
        if not return_hessian_diagonal:
            hess_rows = []
            for i in range(num_classes):
                grad_sum = grad_th[:, i].sum()
                h_row = torch.autograd.grad(grad_sum, y_pred_th, retain_graph=True)[0]
                hess_rows.append(h_row)
            
            hess_th = torch.stack(hess_rows, dim=1)  # (batch, classes, classes)
        
        # Scale for proper normalization
        scale = batch_sz / total_samples if total_samples is not None else 1.0
        grad_th = grad_th * scale
        hess_th = hess_th * scale
        
        return qt.Tensor(grad_data, inds=grad_inds), qt.Tensor(hess_data, inds=hess_inds)
```

---

## 6. Summary Table: Tensor Shapes and Indices

### MPO2 (L=3, bond_dim=4, phys_dim=2, output_dim=3, output_site=2)

| Component | Shape | Indices | Tags |
|---|---|---|---|
| Node0 | (2, 4) | ("x0", "b0") | {"Node0"} |
| Node1 | (4, 2, 4) | ("b0", "x1", "b1") | {"Node1"} |
| Node2 | (4, 2, 3) | ("b1", "x2", "out") | {"Node2"} |
| Input0 | (batch, 2) | ("s", "x0") | {"input_x0", "I0"} |
| Input1 | (batch, 2) | ("s", "x1") | {"input_x1", "I1"} |
| Input2 | (batch, 2) | ("s", "x2") | {"input_x2", "I2"} |
| Output | (batch, 3) | ("s", "out") | {"output"} |

### Environment for Node2

| Component | Shape | Indices |
|---|---|---|
| Environment | (batch, b1, x2, out) | ("s", "b1", "x2", "out") |
| Node2 | (4, 2, 3) | ("b1", "x2", "out") |
| Forward (env & node2) | (batch, 3) | ("s", "out") |

### Gradient/Hessian for Node2

| Component | Shape | Indices |
|---|---|---|
| dL/dy | (batch, 3) | ("s", "out") |
| d²L/dy² | (batch, 3, 3) | ("s", "out", "out_prime") |
| Node Gradient | (4, 2, 3) | ("b1", "x2", "out") |
| Node Hessian | (4, 2, 3, 4, 2, 3) | ("b1", "x2", "out", "b1_prime", "x2_prime", "out_prime") |

---

## 7. Key Design Patterns

### 7.1 Index Naming Conventions

1. **Batch Dimension**: Always "s" (sample)
2. **Physical Indices**: "x{i}" for input features
3. **Bond Indices**: "b{i}" for MPS bonds, "b_mpo_{i}" for MPO bonds, "b_mps_{i}" for MPS bonds in LMPO2
4. **Output**: "out" (single, shared)
5. **Primed Indices**: "{index}_prime" for Hessian second derivatives
6. **Tags**: "Node{i}" for site location, "NT" for non-trainable

### 7.2 Tensor Network Operations

1. **Contraction**: `tn.contract(output_inds=[...])` - contracts all unspecified indices
2. **Fusion**: `tensor.fuse({name: [indices]})` - combines multiple indices into one
3. **Unfusion**: `tensor.unfuse({name: indices}, shape_map={...})` - splits fused index back
4. **Reindexing**: `tn.reindex({old: new})` - renames indices
5. **Deletion**: `tn.delete(tag)` - removes tensor by tag

### 7.3 Regularization

```
L_total = L_data + λ * ||w||²

Newton update with regularization:
(H + 2λI) * Δw = -∇L_data + 2λ * w_old
```

---

## 8. Mathematical Structure Summary

### Forward Pass
```
y = contract(TN ⊗ inputs, output_inds)
```

### Environment-Based Optimization
```
Environment: E = contract(TN \ {node} ⊗ inputs, node_inds ∪ output_inds)
Forward: y = contract(E ⊗ node, output_inds)
Gradient: ∇_node L = contract(E ⊗ ∇_y L, node_inds)
Hessian: H_node = contract(E ⊗ ∇²_y L ⊗ E†, node_inds ⊗ node_inds')
```

### Newton Update
```
Δw = -H^{-1} * ∇L
w_new = w_old + Δw
```

### Regularized Newton Update
```
(H + 2λI) * Δw = -∇L + 2λ * w_old
```

