# NTN Optimization: Side-by-Side Code Comparison

## 1. RIDGE/JITTER APPLICATION

### GTN: Unscaled Ridge (lines 673-692)
```python
if regularize:
    current_node = self.tn[node_tag].copy()
    current_node.fuse(map_b, inplace=True)
    old_weight = current_node.to_dense(["cols"])
    
    scaled_jitter = 2 * effective_jitter  # NO scaling by diagonal
    
    if backend == "torch":
        matrix_data.diagonal().add_(scaled_jitter)
        gradient_vector = gradient_vector + scaled_jitter * old_weight
    elif backend == "numpy":
        rows, cols = lib.diag_indices_from(matrix_data)
        matrix_data[rows, cols] += scaled_jitter
        gradient_vector = gradient_vector + scaled_jitter * old_weight
    elif backend == "jax":
        d_idx = lib.arange(matrix_data.shape[0])
        matrix_data = matrix_data.at[d_idx, d_idx].add(scaled_jitter)
        gradient_vector = gradient_vector + scaled_jitter * old_weight
```

### TNOld: Scaled Ridge (lines 304-316)
```python
elif method.lower() == 'ridge_exact':
    ##A_f.diagonal(dim1=-2, dim2=-1).add_(2 * eps)
    A_f = A_f + (2 * eps) * torch.eye(A_f.shape[-1], dtype=A_f.dtype, device=A_f.device)
    b_f = b_f + (2 * eps) * node.tensor.flatten()
    x = torch.linalg.solve(A_f, -b_f)
elif method.lower().startswith('ridge_cholesky'):
    A_f = A_f + (2 * eps) * torch.eye(A_f.shape[-1], dtype=A_f.dtype, device=A_f.device)
    b_f = b_f + (2 * eps) * node.tensor.flatten()
    L = torch.linalg.cholesky(A_f)
    x = torch.cholesky_solve(-b_f.unsqueeze(-1), L)
    x = x.squeeze(-1)
```

**Key Difference:**
- GTN: `scaled_jitter = 2 * jitter` (constant)
- TNOld: Applied AFTER `A_f = A_f / scale` (adaptive)

---

## 2. SCALING COMPUTATION

### GTN: Computed but Unused (lines 659-669)
```python
backend, lib = self.get_backend(matrix_data)
if backend == "torch":
    scale = matrix_data.diagonal().abs().mean()
    if not torch.isfinite(scale) or scale == 0:
        scale = torch.tensor(1.0, dtype=matrix_data.dtype)
elif backend == "numpy":
    scale = lib.abs(lib.diag(matrix_data)).mean()
    if not lib.isfinite(scale) or scale == 0:
        scale = 1.0
else:
    scale = 1.0

# NOTE: scale is computed but NEVER USED in regularization!
```

### TNOld: Computed and Used (lines 298-302)
```python
A_f = A_f.flatten(0, A_f.ndim//2-1).flatten(1, -1)
b_f = b_f.flatten()
scale = A_f.diag().abs().mean()
if scale == 0:
    scale = 1
A_f = A_f / scale  # ← USED HERE
b_f = b_f / scale  # ← USED HERE
```

**Key Difference:**
- GTN: Computes scale but doesn't use it
- TNOld: Computes and applies scale to normalize system

---

## 3. SWEEP ORDER CONSTRUCTION

### GTN: Asymmetric Sweep (lines 919-923)
```python
if full_sweep_order is None:
    trainable_nodes = self._get_trainable_nodes()
    
    back_sweep = trainable_nodes[-2:0:-1]  # Reverse, exclude first and last
    full_sweep_order = trainable_nodes + back_sweep
```

**Example with 4 nodes:**
```
trainable_nodes = [0, 1, 2, 3]
back_sweep = [2, 1]  # trainable_nodes[-2:0:-1]
full_sweep_order = [0, 1, 2, 3, 2, 1]
```

### TNOld: Symmetric Sweep (lines 418-425, 519-527)
```python
# LEFT TO RIGHT
if node_order is not None:
    if isinstance(node_order, tuple):
        first_node_order = node_order[0]
    else:
        first_node_order = node_order
else:
    first_node_order = self.train_nodes
first_node_order = list(first_node_order if direction == 'l2r' else reversed(first_node_order))

# ... optimize all nodes in first_node_order ...

# RIGHT TO LEFT
if node_order is not None:
    if isinstance(node_order, tuple):
        second_node_order = node_order[1]
    else:
        second_node_order = reversed(node_order)
else:
    second_node_order = self.train_nodes
second_node_order = list(second_node_order if direction == 'r2l' else reversed(list(second_node_order)))
```

**Example with 4 nodes:**
```
first_node_order = [0, 1, 2, 3]   (L2R)
second_node_order = [3, 2, 1, 0]  (R2L)
Full sweep = [0, 1, 2, 3, 3, 2, 1, 0]
```

---

## 4. GRADIENT/HESSIAN COMPUTATION

### GTN: Environment-based (lines 139-188)
```python
def _batch_node_derivatives(self, inputs, y_true, node_tag):
    """Worker for a single batch: Returns (Node_Grad, Node_Hess)"""
    tn = self.tn
    env = self._batch_environment(
        inputs, tn, target_tag=node_tag, sum_over_batch=False, sum_over_output=False
    )
    target_tensor = tn[node_tag]

    y_pred = self.forward_from_environment(
        env, node_tag=node_tag, node_tensor=target_tensor, sum_over_batch=False
    )

    dL_dy, d2L_dy2 = self.loss.get_derivatives(
        y_pred, y_true, backend=self.backend, batch_dim=self.batch_dim,
        output_dims=self.output_dimensions, return_hessian_diagonal=False,
        total_samples=self.train_data.samples,
    )

    # Node Jacobian: J = E * dL_dy
    grad_tn = env & dL_dy
    node_inds = target_tensor.inds
    node_grad = grad_tn.contract(output_inds=node_inds)
    
    # Node Hessian: H = (E * d2L) * E^T
    out_inds = self.output_dimensions
    out_row_inds = out_inds
    out_col_inds = [x + "_prime" for x in out_inds]

    d2L_tensor = qt.Tensor(
        d2L_dy2.data, inds=[self.batch_dim] + out_row_inds + out_col_inds
    )
    env_right = self._prime_indices_tensor(env, exclude_indices=[self.batch_dim])

    hess_tn = env & d2L_tensor & env_right

    node_inds = target_tensor.inds
    hess_out_inds = list(node_inds) + [f"{x}_prime" for x in node_inds]

    node_hess = hess_tn.contract(output_inds=hess_out_inds)

    return node_grad, node_hess
```

### TNOld: Einsum-based (lines 175-217)
```python
@torch.no_grad()
def get_A_b(self, node, grad, hessian, method=None):
    """Finds the update step for a given node"""

    # Determine broadcast
    broadcast_dims = tuple(d for d in self.output_labels if d not in node.dim_labels)
    non_broadcast_dims = tuple(d for d in self.output_labels if d != self.sample_dim)

    # Compute the Jacobian
    J = self.compute_jacobian_stack(node).copy().expand_labels(self.output_labels, grad.shape).permute_first(*broadcast_dims)

    # Assign unique einsum labels
    dim_labels = EinsumLabeler()

    dd_loss_ein = ''.join([dim_labels[self.sample_dim]] + [dim_labels[d] for d in non_broadcast_dims] + [dim_labels['_' + d] for d in non_broadcast_dims])
    d_loss_ein = ''.join(dim_labels[d] for d in self.output_labels)

    J_ein1 = ''
    J_ein2 = ''
    J_out1 = []
    J_out2 = []
    dim_order = []
    for d in J.dim_labels:
        J_ein1 += dim_labels[d]
        J_ein2 += dim_labels['_' + d] if d != self.sample_dim else dim_labels[d]
        if d not in broadcast_dims:
            J_out1.append(dim_labels[d])
            J_out2.append(dim_labels['_' + d])
            dim_order.append(d)
    J_out1 = ''.join([J_out1[dim_order.index(d)] for d in node.dim_labels])
    J_out2 = ''.join([J_out2[dim_order.index(d)] for d in node.dim_labels])

    # Construct einsum notations
    einsum_A = f'{J_ein1},{J_ein2},{dd_loss_ein}->{J_out1}{J_out2}'
    einsum_b = f"{J_ein1},{d_loss_ein}->{J_out1}"

    # Compute einsum operations
    if method is None:
        A = torch.einsum(einsum_A, J.tensor.conj(), J.tensor, hessian)
    else:
        A = torch.randn((2,2,2,2))
    b = torch.einsum(einsum_b, J.tensor.conj(), grad)

    return A, b
```

**Key Differences:**
- GTN: Uses tensor network contraction with environment
- TNOld: Uses einsum with explicit label management
- Both compute: `A = J^H * d²L * J`, `b = J^H * dL`

---

## 5. BATCH ACCUMULATION

### GTN: Inside Function (lines 190-198)
```python
def _compute_H_b(self, node_tag):
    """High-level API to get Jacobian and Hessian for a node using the stored data stream."""
    J, H = self.compute_node_derivatives(
        self.tn, node_tag, self.data.data_mu_y, sum_over_batches=True
    )
    return J, H
```

The accumulation happens inside `compute_node_derivatives` via `_sum_over_batches`.

### TNOld: In Main Loop (lines 438-468)
```python
A_out, b_out = None, None
total_loss = 0.0

for b in tqdm(range(batches), desc=f"Left to right pass ({node_l2r.name if hasattr(node_l2r, 'name') else 'node'})", disable=disable_tqdm):
    # ... get batch data ...
    
    y_pred = self.forward(x_batch, to_tensor=True)
    loss, d_loss, sqd_loss = loss_fn.forward(y_pred, y_batch)
    if method == 'gradient':
        A, b_vec = self.get_A_b(node_l2r, d_loss, sqd_loss, method=method)
    else:
        A, b_vec = self.get_A_b(node_l2r, d_loss, sqd_loss)

    if A_out is None:
        A_out = A
        b_out = b_vec
    else:
        A_out.add_(A)
        b_out.add_(b_vec)
    
    if method == 'gradient':
        node_l2r.update_node(b_vec, lr=lr, adaptive_step=adaptive_step, min_norm=min_norm, max_norm=max_norm)

    total_loss += loss.mean().item()
```

**Key Differences:**
- GTN: Accumulation is abstracted away
- TNOld: Explicit accumulation in loop with `A_out.add_(A)`

---

## 6. MEMORY MANAGEMENT

### GTN: Explicit Cleanup (lines 703-706)
```python
# Clean up large matrices to free memory
del H, b, matrix_data, gradient_vector
if torch.cuda.is_available():
    torch.cuda.empty_cache()
```

Also in `_batch_node_derivatives` (lines 183-186):
```python
# Clean up intermediate tensors to free memory
del env, env_right, d2L_tensor, hess_tn, grad_tn, y_pred, dL_dy, d2L_dy2
if torch.cuda.is_available():
    torch.cuda.empty_cache()
```

### TNOld: Implicit Cleanup
```python
# No explicit cleanup
# Relies on Python garbage collection
```

**Key Difference:**
- GTN: Aggressively frees GPU memory
- TNOld: Relies on automatic garbage collection

---

## 7. SOLVER INTERFACE

### GTN: Simple Interface (lines 548-580)
```python
def solve_linear_system(self, matrix_data, vector_data, method="cholesky"):
    """Solves Ax = b."""
    backend_name, lib = self.get_backend(matrix_data)

    if method == "cholesky":
        result_data = self.cholesky_solve_helper(matrix_data, vector_data, backend_name, lib)
    else:
        if backend_name == "torch":
            b = vector_data
            if b.ndim == matrix_data.ndim - 1:
                b = b.unsqueeze(-1)
            res = lib.linalg.solve(matrix_data, b)
            result_data = res.squeeze(-1) if vector_data.ndim == matrix_data.ndim - 1 else res
        elif backend_name == "numpy":
            result_data = lib.linalg.solve(matrix_data, vector_data)
        elif backend_name == "jax":
            result_data = lib.linalg.solve(matrix_data, vector_data)

    return result_data
```

### TNOld: Flexible Interface (lines 293-327)
```python
def solve_system(self, node, A, b, method='exact', eps=0.0):
    """Finds the update step for a given node"""
    # Solve the system
    A_f = A.flatten(0, A.ndim//2-1).flatten(1, -1)
    b_f = b.flatten()
    scale = A_f.diag().abs().mean()
    if scale == 0:
        scale = 1
    A_f = A_f / scale
    b_f = b_f / scale

    if method.lower() == 'exact':
        x = torch.linalg.solve(A_f, -b_f)
    elif method.lower() == 'ridge_exact':
        A_f = A_f + (2 * eps) * torch.eye(A_f.shape[-1], dtype=A_f.dtype, device=A_f.device)
        b_f = b_f + (2 * eps) * node.tensor.flatten()
        x = torch.linalg.solve(A_f, -b_f)
    elif method.lower().startswith('ridge_cholesky'):
        A_f = A_f + (2 * eps) * torch.eye(A_f.shape[-1], dtype=A_f.dtype, device=A_f.device)
        b_f = b_f + (2 * eps) * node.tensor.flatten()
        L = torch.linalg.cholesky(A_f)
        x = torch.cholesky_solve(-b_f.unsqueeze(-1), L)
        x = x.squeeze(-1)
    elif method.lower() == 'cholesky':
        L = torch.linalg.cholesky(A_f)
        x = torch.cholesky_solve(-b_f.unsqueeze(-1), L)
        x = x.squeeze(-1)
    elif method.lower() == 'gradient':
        x = -b
    else:
        raise ValueError(f"Unknown method: {method}")

    step_tensor = x.reshape(b.shape)
    return step_tensor
```

**Key Differences:**
- GTN: 2 methods (cholesky, standard)
- TNOld: 5 methods (exact, ridge_exact, ridge_cholesky, cholesky, gradient)

---

## Summary: Key Algorithmic Differences

| Aspect | GTN | TNOld |
|--------|-----|-------|
| **Ridge Scaling** | Unscaled (computed but unused) | Scaled by diagonal mean |
| **Regularization** | Fixed: `2*jitter` | Adaptive: `2*eps/scale` |
| **Sweep Pattern** | Asymmetric: [0,1,2,3,2,1] | Symmetric: [0,1,2,3,3,2,1,0] |
| **Grad/Hess Computation** | Environment contraction | Einsum operations |
| **Batch Accumulation** | Inside `_sum_over_batches` | Explicit in loop |
| **Memory Management** | Explicit `torch.cuda.empty_cache()` | Implicit GC |
| **Solver Methods** | 2 (cholesky, standard) | 5 (exact, ridge_exact, ridge_cholesky, cholesky, gradient) |
| **Numerical Stability** | Lower (unscaled) | Higher (scaled) |

