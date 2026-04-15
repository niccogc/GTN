# NTN Optimization Analysis: Complete Index

This directory contains a comprehensive comparison of NTN (Newton Tensor Network) optimization between GTN and TNOld implementations.

## 📋 Documents Overview

### 1. **NTN_SUMMARY.txt** ⭐ START HERE
   - **Purpose**: Executive summary with key findings
   - **Length**: ~2 pages
   - **Best for**: Quick understanding of main differences
   - **Key sections**:
     - Critical algorithmic differences
     - Performance trade-offs
     - When to use which implementation
     - Recommended hybrid approach

### 2. **NTN_COMPARISON.md**
   - **Purpose**: Detailed technical comparison
   - **Length**: ~4 pages
   - **Best for**: Understanding the full picture
   - **Key sections**:
     - Ridge/jitter application (most important!)
     - Gradient/Hessian computation
     - Sweep order and direction
     - Regularization formulation
     - Mathematical equivalence
     - Practical implications
     - Convergence behavior

### 3. **NTN_DETAILED_ANALYSIS.md**
   - **Purpose**: Deep dive into specific differences
   - **Length**: ~5 pages
   - **Best for**: Understanding why differences matter
   - **Key sections**:
     - Critical finding: Scaling difference
     - Sweep order visualization
     - Regularization strength comparison
     - Batch accumulation strategy
     - Solver method comparison
     - Memory management
     - Detailed comparison table
     - Recommendations

### 4. **NTN_CODE_COMPARISON.md**
   - **Purpose**: Side-by-side code snippets
   - **Length**: ~6 pages
   - **Best for**: Developers who want to see actual code
   - **Key sections**:
     - Ridge/jitter application code
     - Scaling computation code
     - Sweep order construction code
     - Gradient/Hessian computation code
     - Batch accumulation code
     - Memory management code
     - Solver interface code
     - Summary comparison table

### 5. **NTN_VISUAL_COMPARISON.md**
   - **Purpose**: Visual diagrams and flowcharts
   - **Length**: ~4 pages
   - **Best for**: Visual learners
   - **Key sections**:
     - Ridge/jitter application flow
     - Sweep order comparison diagrams
     - Regularization strength visualization
     - Numerical stability comparison
     - Convergence behavior diagrams
     - Memory management graphs
     - Decision tree
     - Summary matrix

## 🎯 Quick Navigation

### I want to understand...

**The main differences:**
→ Read: NTN_SUMMARY.txt (2 min)

**Why GTN and TNOld differ:**
→ Read: NTN_COMPARISON.md (10 min)

**The technical details:**
→ Read: NTN_DETAILED_ANALYSIS.md (15 min)

**The actual code:**
→ Read: NTN_CODE_COMPARISON.md (20 min)

**Visual explanations:**
→ Read: NTN_VISUAL_COMPARISON.md (10 min)

**Everything:**
→ Read all documents in order (60 min)

## 🔑 Key Findings

### 1. Ridge/Jitter Application (MOST IMPORTANT)
- **GTN**: Unscaled ridge → Less stable with ill-conditioned Hessians
- **TNOld**: Scaled ridge → More stable with ill-conditioned Hessians

### 2. Sweep Order
- **GTN**: Asymmetric (N + N-1 updates) → Faster per-epoch
- **TNOld**: Symmetric (2N updates) → More balanced optimization

### 3. Regularization Strength
- **GTN**: Fixed (2*jitter) → Same strength regardless of Hessian
- **TNOld**: Adaptive (2*eps/scale) → Scales with Hessian magnitude

### 4. Memory Management
- **GTN**: Explicit cleanup → Better for memory-constrained GPUs
- **TNOld**: Implicit cleanup → Relies on garbage collection

### 5. Solver Flexibility
- **GTN**: 2 methods (cholesky, standard)
- **TNOld**: 5 methods (exact, ridge_exact, ridge_cholesky, cholesky, gradient)

## 📊 Comparison Matrix

| Feature | GTN | TNOld |
|---------|-----|-------|
| **Ridge Scaling** | ❌ No | ✅ Yes |
| **Regularization** | Fixed | Adaptive |
| **Sweep Pattern** | Asymmetric | Symmetric |
| **Memory Mgmt** | Explicit | Implicit |
| **Solver Methods** | 2 | 5 |
| **Numerical Stability** | Lower | Higher |
| **Speed/Epoch** | Faster | Slower |
| **Robustness** | Lower | Higher |

## 🚀 Recommendations

### Use GTN when:
- ✓ Hessian is well-conditioned
- ✓ Memory is extremely limited
- ✓ You want fastest per-epoch convergence
- ✓ Working with MPS-like structures
- ✓ You need explicit GPU memory control

### Use TNOld when:
- ✓ Hessian may be ill-conditioned
- ✓ Numerical stability is critical
- ✓ You want symmetric treatment of all nodes
- ✓ You need flexible solver options
- ✓ You want adaptive regularization strength

### Hybrid Approach:
Combine GTN's explicit memory management with TNOld's scaled regularization:
```python
# Compute scale
scale = matrix_data.diagonal().abs().mean()
if scale == 0:
    scale = 1.0

# Scale the regularization
scaled_jitter = 2 * effective_jitter / scale

# Apply to diagonal
matrix_data.diagonal().add_(scaled_jitter)
gradient_vector = gradient_vector + scaled_jitter * old_weight

# Explicit cleanup
del H, b, matrix_data, gradient_vector
if torch.cuda.is_available():
    torch.cuda.empty_cache()
```

## 📁 File Locations

- **GTN Implementation**: `/home/nicci/Desktop/remote/GTN/model/base/NTN.py`
- **TNOld Implementation**: `/home/nicci/Desktop/remote/TNOld/tensor/network.py`

## 🔍 Key Code Locations

### GTN
- Ridge application: Lines 673-692
- Scaling computation: Lines 659-669
- Sweep order: Lines 919-923
- Gradient/Hessian: Lines 139-188
- Memory cleanup: Lines 183-186, 703-706

### TNOld
- Ridge application: Lines 304-316
- Scaling computation: Lines 298-302
- Sweep order: Lines 418-425, 519-527
- Gradient/Hessian: Lines 175-217
- Batch accumulation: Lines 438-468

## 💡 Key Insight

The **MOST IMPORTANT difference** is the ridge/jitter scaling:

```
GTN:    Unscaled → Less stable with ill-conditioned Hessians
TNOld:  Scaled   → More stable with ill-conditioned Hessians
```

For production use with potentially ill-conditioned problems, TNOld's approach is recommended. For well-conditioned problems with memory constraints, GTN's approach is sufficient.

## 📚 Reading Order

1. **NTN_SUMMARY.txt** (2 min) - Get the overview
2. **NTN_VISUAL_COMPARISON.md** (10 min) - See the diagrams
3. **NTN_COMPARISON.md** (10 min) - Understand the details
4. **NTN_DETAILED_ANALYSIS.md** (15 min) - Deep dive
5. **NTN_CODE_COMPARISON.md** (20 min) - See the code

Total time: ~60 minutes for complete understanding

## ✅ Checklist

- [ ] Read NTN_SUMMARY.txt
- [ ] Review NTN_VISUAL_COMPARISON.md
- [ ] Study NTN_COMPARISON.md
- [ ] Understand NTN_DETAILED_ANALYSIS.md
- [ ] Review NTN_CODE_COMPARISON.md
- [ ] Decide which implementation to use
- [ ] Consider hybrid approach if needed

## 🤔 Questions?

Refer to the specific document:
- **"What's the main difference?"** → NTN_SUMMARY.txt
- **"Why does it matter?"** → NTN_DETAILED_ANALYSIS.md
- **"Show me the code"** → NTN_CODE_COMPARISON.md
- **"Draw me a picture"** → NTN_VISUAL_COMPARISON.md
- **"Tell me everything"** → NTN_COMPARISON.md

---

**Last Updated**: 2024-04-15
**Analysis Scope**: GTN vs TNOld NTN Optimization
**Files Analyzed**: 
- GTN/model/base/NTN.py (1096 lines)
- TNOld/tensor/network.py (1060 lines)

