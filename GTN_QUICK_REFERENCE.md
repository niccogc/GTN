# GTN Architecture - Quick Reference Guide

## 1. Tensor Shapes at a Glance

### MPO2 (L=3, bond_dim=4, phys_dim=2, output_dim=3)

```
Node0: (2, 4)       x0 --[Node0]-- b0
Node1: (4, 2, 4)    b0 --[Node1]-- b1
                         x1
Node2: (4, 2, 3)    b1 --[Node2]-- out
                         x2
```

### Index Patterns

| Type | Pattern | Example |
|------|---------|---------|
| Physical | `x{i}` | x0, x1, x2 |
| Bond | `b{i}` | b0, b1 |
| Output | `out` | out |
| Batch | `s` | s |
| Primed | `{idx}_prime` | b0_prime, x1_prime |

---

## 2. Forward Pass Flow

```
Input Data (batch, features)
    ↓
create_inputs() → Inputs object
    ↓
_batch_forward(inputs, tn, output_inds)
    ↓
full_tn = tn & inputs  (contract inputs into network)
    ↓
res = full_tn.contract(output_inds=["s", "out"])
    ↓
Output (batch, output_dim)
```

---

## 3. Environment-Based Optimization

```
For each node:
  1. env = _batch_environment(inputs, tn, target_tag)
     → All network except target node
     → Shape: (batch, *node_inds, *output_inds)
  
  2. y_pred = forward_from_environment(env, node_tensor)
     → Fast forward pass using pre-computed env
  
  3. dL_dy, d2L_dy2 = loss.get_derivatives(y_pred, y_true)
     → Gradient and Hessian w.r.t. output
  
  4. node_grad = contract(env & dL_dy, output_inds=node_inds)
     → Gradient w.r.t. node parameters
  
  5. node_hess = contract(env & d2L_dy2 & env_prime, 
                          output_inds=node_inds + node_inds_prime)
     → Hessian w.r.t. node parameters
  
  6. Δw = -H^{-1} * ∇L  (Newton update)
     → Solve linear system
```

---

## 4. Initialization Parameters

### MPO2 Constructor

```python
MPO2(
    L=3,                          # Number of sites
    bond_dim=4,                   # Bond dimension
    phys_dim=2,                   # Physical dimension per site
    output_dim=3,                 # Output dimension
    output_site=2,                # Which site gets output (default: L-1)
    init_strength=0.001,          # Only used if use_tn_normalization=False
    use_tn_normalization=True,    # Apply normalization after init
    tn_target_std=0.1,            # Target output std (if sample_inputs provided)
    sample_inputs=None            # Sample data for output-based normalization
)
```

### Initialization Flow

```
1. base_init = 0.1 if use_tn_normalization else init_strength
2. Create tensors: data = torch.randn(*shape) * base_init
3. If use_tn_normalization:
   - If sample_inputs: normalize_tn_output(tn, sample_inputs, target_std=0.1)
   - Else: normalize_tn_frobenius(tn, target_norm=√(L·bond_dim·phys_dim))
```

---

## 5. Input Preparation

### create_inputs() Function

```python
create_inputs(
    X,                    # (samples, features)
    y,                    # (samples,) or (samples, output_dim)
    input_labels=None,    # List of index names for each input
    output_labels=None,   # Default: ["out"]
    batch_size=32,
    batch_dim="s",
    append_bias=True,     # KEY: Appends column of ones to X
    encoding=None,        # "polynomial" or "fourier"
    poly_degree=None
)
```

### Bias Term

```python
if append_bias:
    X = torch.cat([X, torch.ones(n_samples, 1)], dim=1)
    # X shape: (samples, features+1)
    # Last column is all ones
```

### Polynomial Encoding

```python
encode_polynomial(X, degree=2)
# Input:  (samples, features)
# Output: (samples, features, degree+1)
# For each feature x_i: [1, x_i, x_i^2, ..., x_i^degree]
```

---

## 6. Loss Functions

### MSELoss (Regression)

```python
use_diagonal_hessian = True

Gradient: dL/dy = 2 * (y_pred - y_true)
Hessian:  d²L/dy² = 2 (constant, diagonal)
```

### CrossEntropyLoss (Classification)

```python
use_diagonal_hessian = False  # FULL Hessian required!

Gradient: dL/dz_i = p_i - y_i  (p = softmax(logits))
Hessian:  d²L/dz_i dz_j = p_i * (δ_ij - p_j)  (FULL MATRIX)
```

---

## 7. Training Loop

### Algorithm Selection

```python
if isinstance(loss, (MSELoss, MAELoss, HuberLoss)):
    use_lstsq = True  # Direct least squares solver
else:
    use_lstsq = False  # Newton-based optimization
```

### Sweep Order

```python
trainable_nodes = [Node0, Node1, Node2]
back_sweep = [Node1, Node0]  # Reverse, excluding first and last
full_sweep_order = [Node0, Node1, Node2, Node1, Node0]
```

### Regularization

```python
# L2 regularization: minimize ||w||² + λ * ||w - w_old||²
# Equivalent to: (H + 2λI) * Δw = -∇L + 2λ * w_old

scaled_jitter = 2 * jitter
H_reg = H + scaled_jitter * I
b_reg = -b + scaled_jitter * w_old
```

---

## 8. Key Tensor Operations

### Contraction

```python
# Contract all unspecified indices
result = tn.contract(output_inds=["s", "out"])
# Remaining indices: s, out
# All others contracted (summed over)
```

### Fusion

```python
# Combine multiple indices into one
tensor.fuse({"cols": ["b0", "x1", "out"]}, inplace=True)
# Shape: (batch, cols) where cols = b0 × x1 × out
```

### Unfusion

```python
# Split fused index back
tensor.unfuse(
    {"cols": ["b0", "x1", "out"]},
    shape_map={"cols": (4, 2, 3)},
    inplace=True
)
# Shape: (batch, 4, 2, 3)
```

---

## 9. Non-Trainable Tensors

### Marking Non-Trainable

```python
# Add "NT" tag to tensor
tensor.add_tag("NT")

# Or in constructor
tensor = qt.Tensor(data, inds=inds, tags={"Node0", "NT"})
```

### Effect

- Excluded from optimization
- Not included in regularization
- Preserved during training

### Example: MMPO2

```python
# Mask tensors are non-trainable
mask_tensor = qt.Tensor(data, inds=inds, tags={"0_Mask", "NT"})

# MPS tensors are trainable
mps_tensor = qt.Tensor(data, inds=inds, tags={"0_MPS"})
```

---

## 10. Common Patterns

### Pattern 1: Single Input with Bias

```python
X = torch.randn(100, 5)  # 100 samples, 5 features
y = torch.randn(100, 1)  # 100 targets

inputs = create_inputs(X, y, input_labels=["x"], append_bias=True)
# X becomes (100, 6) with last column = 1
```

### Pattern 2: Multiple Inputs (CMPO2)

```python
X_pixels = torch.randn(100, 10)
X_patches = torch.randn(100, 10)

inputs = Inputs(
    inputs=[X_patches, X_pixels],
    outputs=[y],
    input_labels=[[0, ("0_patches", "0_pixels")], ...],
    batch_dim="s"
)
```

### Pattern 3: Polynomial Features

```python
X = torch.randn(100, 3)
inputs = create_inputs(X, y, encoding="polynomial", poly_degree=2)
# X becomes (100, 3, 3) with [1, x_i, x_i^2] for each feature
```

---

## 11. Debugging Checklist

- [ ] Check tensor shapes match indices
- [ ] Verify batch_dim is consistent ("s")
- [ ] Ensure output_dims match loss function
- [ ] Check for index conflicts (duplicate names)
- [ ] Verify non-trainable tags ("NT") are correct
- [ ] Confirm initialization scale (base_init)
- [ ] Check regularization jitter value
- [ ] Verify sweep order includes all trainable nodes

---

## 12. Performance Tips

1. **Environment Caching**: Compute environment once, reuse for multiple forward passes
2. **Batch Processing**: Use batches to amortize environment computation
3. **Regularization**: Use jitter > 0 to avoid singular matrices
4. **Early Stopping**: Set patience to avoid overfitting
5. **Normalization**: Use output-based normalization for better initialization

