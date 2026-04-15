# Detailed NTN Optimization Analysis

## CRITICAL FINDING: Scaling Difference

### The Unscaled vs Scaled Ridge Problem

**GTN's Approach (UNSCALED):**
```
1. Compute H (Hessian), b (gradient)
2. Compute scale = mean(|diag(H)|)  ← COMPUTED BUT IGNORED!
3. Apply ridge: H' = H + 2λI
4. Solve: (H + 2λI) Δw = -(b + 2λ*w_old)
```

**TNOld's Approach (SCALED):**
```
1. Compute A (Hessian), b (gradient)
2. Compute scale = mean(|diag(A)|)
3. Normalize: A' = A/scale, b' = b/scale
4. Apply ridge: A'' = A' + 2ε*I
5. Solve: (A/scale + 2ε*I) Δw = -(b/scale + 2ε*w_old)
```

### Why This Matters

**Example: Ill-conditioned Hessian**
```
H = [1000    0  ]    scale = 500
    [   0    0.1]

GTN:  H + 2λI = [1000+2λ    0      ]
                [   0    0.1+2λ   ]
      
      If λ=1e-6: [1000.000002    0        ]
                 [   0         0.100002  ]
      
      Condition number ≈ 10,000 (STILL ILL-CONDITIONED!)

TNOld: (H/500 + 2ε*I) = [2+2ε    0    ]
                        [  0   0.0002+2ε]
       
       If ε=1e-6: [2.000002    0      ]
                  [  0      0.0002002]
       
       Condition number ≈ 10,000 (SAME, but better numerical behavior)
```

**Key Insight:** TNOld's scaling makes the regularization strength **adaptive** to the Hessian magnitude, while GTN uses a **fixed** regularization strength regardless of conditioning.

---

## Sweep Order Visualization

### GTN Sweep Pattern (N=4 nodes)

```
Epoch 1:
  Forward:  [0] → [1] → [2] → [3]
  Backward: [2] → [1]
  ─────────────────────────────────
  Total updates: Node0(1x), Node1(2x), Node2(2x), Node3(1x)
  
  Asymmetric: First and last nodes updated once, middle nodes twice
```

### TNOld Sweep Pattern (N=4 nodes)

```
Sweep 1:
  L2R: [0] → [1] → [2] → [3]
  R2L: [3] → [2] → [1] → [0]
  ─────────────────────────────────
  Total updates: Node0(2x), Node1(2x), Node2(2x), Node3(2x)
  
  Symmetric: All nodes updated equally
```

### Convergence Implications

**GTN's asymmetric pattern:**
- Boundary nodes (first/last) get less optimization
- May be intentional for MPS-like structures
- Faster per-epoch (fewer updates)

**TNOld's symmetric pattern:**
- All nodes treated equally
- Better for general tensor networks
- More thorough optimization per epoch

---

## Regularization Strength Comparison

### Fixed vs Adaptive Regularization

**GTN (Fixed):**
```python
scaled_jitter = 2 * jitter  # Always 2*jitter, regardless of H
```

**TNOld (Adaptive):**
```python
scale = A_f.diag().abs().mean()
# Effective regularization = 2*eps / scale
# Stronger when H is small, weaker when H is large
```

### Example with Different Hessian Magnitudes

```
Scenario 1: Large Hessian (scale=1000)
  GTN:    λ_eff = 2*1e-6 = 2e-6
  TNOld:  λ_eff = 2*1e-6 / 1000 = 2e-9  (much weaker)

Scenario 2: Small Hessian (scale=0.001)
  GTN:    λ_eff = 2*1e-6 = 2e-6
  TNOld:  λ_eff = 2*1e-6 / 0.001 = 2e-3  (much stronger)
```

**Interpretation:**
- GTN: "Always regularize by the same amount"
- TNOld: "Regularize proportionally to Hessian magnitude"

---

## Batch Accumulation Strategy

### GTN: Sum-Over-Batches
```python
def _compute_H_b(self, node_tag):
    J, H = self.compute_node_derivatives(
        self.tn, node_tag, self.data.data_mu_y, 
        sum_over_batches=True  # ← Sums inside
    )
    return J, H
```

**Flow:**
```
For each batch:
  - Compute J_batch, H_batch
  - Accumulate: J_total += J_batch, H_total += H_batch
Return J_total, H_total
```

### TNOld: Accumulate-in-Loop
```python
for b in range(batches):
    A, b_vec = self.get_A_b(node_l2r, d_loss, sqd_loss)
    if A_out is None:
        A_out = A
        b_out = b_vec
    else:
        A_out.add_(A)  # ← Accumulate in place
        b_out.add_(b_vec)
```

**Flow:**
```
A_out = None
For each batch:
  - Compute A_batch, b_batch
  - If first: A_out = A_batch, b_out = b_batch
  - Else: A_out += A_batch, b_out += b_batch
Return A_out, b_out
```

**Difference:**
- GTN: Accumulation happens inside `compute_node_derivatives`
- TNOld: Accumulation happens in the main loop
- Both are mathematically equivalent, different code organization

---

## Solver Method Comparison

### GTN (`solve_linear_system`, lines 548-580)
```python
if method == "cholesky":
    result_data = self.cholesky_solve_helper(matrix_data, vector_data, ...)
else:
    # Standard solve
    result_data = lib.linalg.solve(matrix_data, vector_data)
```

**Supports:**
- Cholesky decomposition (faster, requires positive definite)
- Standard solve (slower, more robust)

### TNOld (`solve_system`, lines 293-327)
```python
if method.lower() == 'exact':
    x = torch.linalg.solve(A_f, -b_f)
elif method.lower() == 'ridge_exact':
    A_f = A_f + (2 * eps) * torch.eye(...)
    x = torch.linalg.solve(A_f, -b_f)
elif method.lower().startswith('ridge_cholesky'):
    A_f = A_f + (2 * eps) * torch.eye(...)
    L = torch.linalg.cholesky(A_f)
    x = torch.cholesky_solve(-b_f.unsqueeze(-1), L)
elif method.lower() == 'cholesky':
    L = torch.linalg.cholesky(A_f)
    x = torch.cholesky_solve(-b_f.unsqueeze(-1), L)
elif method.lower() == 'gradient':
    x = -b  # Gradient descent step
```

**Supports:**
- Exact solve
- Ridge exact (with regularization)
- Ridge Cholesky (with regularization)
- Cholesky
- Gradient descent (fallback)

**TNOld is more flexible** with explicit ridge variants.

---

## Memory Management

### GTN (Explicit)
```python
# Clean up large matrices to free memory
del H, b, matrix_data, gradient_vector
if torch.cuda.is_available():
    torch.cuda.empty_cache()  # ← Force GPU cleanup
```

### TNOld (Implicit)
```python
# No explicit cleanup
# Relies on Python garbage collection
```

**GTN is more aggressive** about freeing GPU memory, which can help with large models.

---

## Summary Table: Detailed Comparison

| Feature | GTN | TNOld |
|---------|-----|-------|
| **Ridge Scaling** | ❌ Unscaled | ✅ Scaled by diag mean |
| **Regularization Strength** | Fixed | Adaptive |
| **Sweep Symmetry** | Asymmetric (N + N-1) | Symmetric (2N) |
| **Boundary Node Updates** | 1x | 2x |
| **Interior Node Updates** | 2x | 2x |
| **Batch Accumulation** | Inside function | In main loop |
| **Solver Flexibility** | 2 methods | 5 methods |
| **Memory Management** | Explicit cleanup | Implicit GC |
| **Numerical Stability** | Lower (unscaled) | Higher (scaled) |
| **Convergence Speed** | Potentially faster | Potentially slower |
| **Robustness** | Lower | Higher |

---

## Recommendations

### Use GTN when:
- Hessian is well-conditioned
- Memory is extremely limited (explicit cleanup helps)
- You want faster per-epoch convergence
- Working with MPS-like structures (asymmetric sweep is intentional)

### Use TNOld when:
- Hessian may be ill-conditioned
- Numerical stability is critical
- You want symmetric treatment of all nodes
- You need flexible solver options
- You want adaptive regularization

### Hybrid Approach:
Could combine GTN's explicit memory management with TNOld's scaled regularization:
```python
scaled_jitter = 2 * effective_jitter / scale  # Scale the regularization
matrix_data.diagonal().add_(scaled_jitter)
gradient_vector = gradient_vector + scaled_jitter * old_weight
# ... then explicit cleanup
```

