# NTN Optimization Comparison: GTN vs TNOld

## Overview
Both implementations use alternating least squares (ALS) / Newton sweeps to optimize tensor networks, but they differ significantly in:
1. **Ridge/Jitter Application Strategy**
2. **Sweep Order and Direction**
3. **Gradient/Hessian Computation**
4. **Regularization Formulation**

---

## 1. RIDGE/JITTER APPLICATION

### GTN (`_get_node_update`, lines 673-692)
```python
if regularize:
    current_node = self.tn[node_tag].copy()
    current_node.fuse(map_b, inplace=True)
    old_weight = current_node.to_dense(["cols"])
    
    scaled_jitter = 2 * effective_jitter  # NO scale factor
    
    # Add to diagonal
    matrix_data.diagonal().add_(scaled_jitter)
    # Add to gradient
    gradient_vector = gradient_vector + scaled_jitter * old_weight
```

**Key Points:**
- Adds `2 * jitter` to Hessian diagonal
- Modifies gradient: `b' = b + 2*jitter*w_old`
- Solves: `(H + 2*jitter*I) * delta = -b'`
- This is **L2 regularization** with weight decay term
- **No scaling by diagonal magnitude**

### TNOld (`solve_system`, lines 293-327)
```python
if method.lower() == 'ridge_exact':
    A_f = A_f + (2 * eps) * torch.eye(A_f.shape[-1], dtype=A_f.dtype, device=A_f.device)
    b_f = b_f + (2 * eps) * node.tensor.flatten()
    x = torch.linalg.solve(A_f, -b_f)
```

**Key Points:**
- Also adds `2 * eps` to diagonal
- Also modifies gradient: `b' = b + 2*eps*w_old`
- **Identical formulation to GTN**
- But applied AFTER scaling by diagonal mean

### Scaling Difference (Critical!)

**GTN** (lines 659-669):
```python
scale = matrix_data.diagonal().abs().mean()
if not torch.isfinite(scale) or scale == 0:
    scale = torch.tensor(1.0, dtype=matrix_data.dtype)
# NOTE: scale is computed but NOT used in regularization!
```

**TNOld** (lines 298-302):
```python
A_f = A_f / scale
b_f = b_f / scale
# Then ridge is applied to scaled system
A_f = A_f + (2 * eps) * torch.eye(...)
b_f = b_f + (2 * eps) * node.tensor.flatten()
```

**DIFFERENCE:** TNOld applies ridge to **scaled** system, GTN applies to **unscaled** system.

---

## 2. GRADIENT/HESSIAN COMPUTATION

### GTN (`_batch_node_derivatives`, lines 139-188)
```python
# Compute environment E (batch, out, node_bonds)
env = self._batch_environment(inputs, tn, target_tag=node_tag, 
                              sum_over_batch=False, sum_over_output=False)

# Forward pass via environment
y_pred = self.forward_from_environment(env, node_tag=node_tag, 
                                       node_tensor=target_tensor, 
                                       sum_over_batch=False)

# Loss derivatives
dL_dy, d2L_dy2 = self.loss.get_derivatives(y_pred, y_true, ...)

# Node Jacobian: J = E * dL_dy
grad_tn = env & dL_dy
node_grad = grad_tn.contract(output_inds=node_inds)

# Node Hessian: H = (E * d2L) * E^T
d2L_tensor = qt.Tensor(d2L_dy2.data, inds=[batch_dim] + out_row_inds + out_col_inds)
env_right = self._prime_indices_tensor(env, exclude_indices=[batch_dim])
hess_tn = env & d2L_tensor & env_right
node_hess = hess_tn.contract(output_inds=hess_out_inds)
```

**Key Points:**
- Computes **full Hessian** (not diagonal)
- Uses environment contraction: `H = E ⊗ d²L ⊗ E^T`
- Sums over batches: `H_total = Σ_batch H_batch`
- Returns: `(gradient, hessian)` tuple

### TNOld (`get_A_b`, lines 175-217)
```python
# Compute Jacobian stack
J = self.compute_jacobian_stack(node).copy().expand_labels(...)

# Construct einsum notations
einsum_A = f'{J_ein1},{J_ein2},{dd_loss_ein}->{J_out1}{J_out2}'
einsum_b = f"{J_ein1},{d_loss_ein}->{J_out1}"

# Compute via einsum
A = torch.einsum(einsum_A, J.tensor.conj(), J.tensor, hessian)
b = torch.einsum(einsum_b, J.tensor.conj(), grad)
```

**Key Points:**
- Also computes **full Hessian** via einsum
- Formula: `A = J^H * d²L * J` (Hessian of loss contracted with Jacobian)
- Accumulates over batches in `accumulating_swipe`
- Returns: `(A, b)` tuple

---

## 3. SWEEP ORDER

### GTN (`fit`, lines 919-923)
```python
if full_sweep_order is None:
    trainable_nodes = self._get_trainable_nodes()
    
    back_sweep = trainable_nodes[-2:0:-1]  # Reverse, exclude first and last
    full_sweep_order = trainable_nodes + back_sweep
```

**Sweep Pattern:**
```
Forward:  [Node0, Node1, Node2, ..., NodeN]
Backward: [NodeN-1, NodeN-2, ..., Node1]
Full:     [Node0, Node1, ..., NodeN, NodeN-1, ..., Node1]
```
- **Bidirectional sweep** (forward + backward)
- Excludes first node in backward pass
- One complete sweep per epoch

### TNOld (`accumulating_swipe`, lines 418-606)
```python
# LEFT TO RIGHT
first_node_order = list(first_node_order if direction == 'l2r' else reversed(first_node_order))
for node_i, node_l2r in enumerate(first_node_order):
    # ... optimize node_l2r ...

# RIGHT TO LEFT (unless skip_second=True)
second_node_order = list(second_node_order if direction == 'r2l' else reversed(list(second_node_order)))
for node_i, node_r2l in enumerate(second_node_order):
    # ... optimize node_r2l ...
```

**Sweep Pattern:**
```
Left-to-Right:  [Node0, Node1, Node2, ..., NodeN]
Right-to-Left:  [NodeN, NodeN-1, ..., Node0]
Full:           [Node0, Node1, ..., NodeN, NodeN, NodeN-1, ..., Node0]
```
- **Full bidirectional sweep** (includes all nodes both directions)
- Can skip second pass with `skip_second=True`
- Multiple sweeps per epoch via `num_swipes`

---

## 4. REGULARIZATION FORMULATION

### GTN: L2 Weight Decay
```
Objective: min ||y - f(w)||² + λ||w||²

Update step solves:
(H + 2λI) * Δw = -(∇L + 2λ*w_old)

Where:
- H = Hessian of loss
- ∇L = gradient of loss
- w_old = current weights
```

**Interpretation:** Adds penalty term `2λ*w_old` to gradient, encouraging weights toward zero.

### TNOld: Identical (after scaling)
```
Same formulation, but applied to scaled system:
(H/scale + 2ε*I) * Δw = -(∇L/scale + 2ε*w_old)
```

---

## 5. KEY ALGORITHMIC DIFFERENCES

| Aspect | GTN | TNOld |
|--------|-----|-------|
| **Ridge Application** | Direct to unscaled system | Applied after diagonal scaling |
| **Scaling Factor** | Computed but unused | Used to normalize system |
| **Hessian Type** | Full Hessian | Full Hessian |
| **Sweep Direction** | Forward + Backward (N + N-1 nodes) | L2R + R2L (2N nodes) |
| **Batch Accumulation** | Sum over batches in `_compute_H_b` | Accumulate A, b in loop |
| **Regularization** | L2 weight decay | L2 weight decay (scaled) |
| **Solver** | Cholesky or standard solve | Cholesky or ridge_exact |
| **Memory Management** | Explicit cleanup with torch.cuda.empty_cache() | Implicit via Python GC |

---

## 6. MATHEMATICAL EQUIVALENCE

**GTN and TNOld solve equivalent problems:**

GTN: `(H + 2λI) Δw = -(∇L + 2λ*w_old)`

TNOld (unscaled): Same equation

TNOld (scaled): `(H/s + 2ε*I) Δw = -(∇L/s + 2ε*w_old)`
- Where `s = mean(|diag(H)|)`
- Equivalent to GTN with `λ = ε*s`

**However, the scaling in TNOld:**
- Improves numerical stability
- Makes regularization strength relative to Hessian magnitude
- GTN's unscaled approach may be more sensitive to Hessian scale

---

## 7. PRACTICAL IMPLICATIONS

### GTN Advantages:
1. **Simpler regularization** - direct L2 weight decay
2. **Explicit memory management** - GPU cache clearing
3. **Cleaner sweep order** - avoids redundant first node update

### TNOld Advantages:
1. **Better numerical stability** - scales system before regularization
2. **More flexible** - supports multiple sweeps per epoch
3. **Adaptive regularization** - strength scales with Hessian magnitude
4. **Full bidirectional sweeps** - potentially better convergence

### Potential Issues:
- **GTN**: Unscaled regularization may cause issues with ill-conditioned Hessians
- **TNOld**: Scaling adds computational overhead but improves robustness

---

## 8. CONVERGENCE BEHAVIOR

**GTN:**
- Fixed regularization strength per epoch
- Bidirectional sweep with N + (N-1) node updates
- May converge faster but less stable with poor conditioning

**TNOld:**
- Adaptive regularization (scaled by Hessian)
- Full bidirectional sweep with 2N node updates
- More stable but potentially slower convergence

