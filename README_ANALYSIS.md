# GTN Model Architecture Analysis - Complete Documentation

This directory contains a comprehensive analysis of the GTN (Gradient Tensor Network) model architecture, focusing on tensor construction, training mechanisms, and mathematical structure.

## 📄 Documentation Files

### 1. **ANALYSIS_SUMMARY.txt** (9.7 KB)
**Start here for a quick overview**
- Executive summary of all components
- Key findings organized by topic
- Exact tensor shapes with examples
- Index naming conventions
- Common pitfalls and solutions
- Files analyzed summary

### 2. **GTN_ARCHITECTURE_ANALYSIS.md** (27 KB)
**Complete technical reference**
- Detailed breakdown of all 4 MPO2 model classes (MPO2, CMPO2, LMPO2, MMPO2)
- Tensor construction with exact shapes and indices
- NTN training engine with forward pass, environment computation, gradient/Hessian
- Input preparation with bias term and feature encodings
- Initialization strategies (output-based and Frobenius normalization)
- Loss functions and their derivatives
- Summary tables with tensor shapes
- Mathematical structure and formulations

### 3. **GTN_QUICK_REFERENCE.md** (7.1 KB)
**Quick lookup guide for common tasks**
- Tensor shapes at a glance
- Forward pass flow diagram
- Environment-based optimization steps
- Initialization parameters
- Input preparation patterns
- Loss function specifications
- Training loop algorithm
- Key tensor operations (contraction, fusion, unfusion)
- Non-trainable tensor handling
- Common patterns and debugging checklist
- Performance tips

### 4. **GTN_DETAILED_EXAMPLES.md** (13 KB)
**10 complete working examples with code**
1. Simple MPO2 Regression
2. CMPO2 with Cross-Connection
3. LMPO2 with Dimensionality Reduction
4. MMPO2 with Non-Trainable Mask
5. Classification with CrossEntropyLoss
6. Polynomial Features
7. Fourier Features
8. Early Stopping with Validation
9. Custom Regularization Schedule
10. Debugging Tensor Shapes

Each example includes:
- Complete setup code
- Tensor shapes during training
- Index structures
- Forward pass details

## 🎯 How to Use This Documentation

### For Quick Understanding
1. Read **ANALYSIS_SUMMARY.txt** (5 min)
2. Skim **GTN_QUICK_REFERENCE.md** (10 min)
3. Look up specific examples in **GTN_DETAILED_EXAMPLES.md**

### For Deep Understanding
1. Start with **ANALYSIS_SUMMARY.txt** for overview
2. Read **GTN_ARCHITECTURE_ANALYSIS.md** section by section
3. Reference **GTN_QUICK_REFERENCE.md** for specific patterns
4. Study **GTN_DETAILED_EXAMPLES.md** for practical implementation

### For Implementation
1. Find relevant example in **GTN_DETAILED_EXAMPLES.md**
2. Check tensor shapes in **GTN_QUICK_REFERENCE.md**
3. Reference exact formulations in **GTN_ARCHITECTURE_ANALYSIS.md**
4. Debug using checklist in **GTN_QUICK_REFERENCE.md**

## 📊 Key Concepts at a Glance

### Tensor Shapes (L=3, bond_dim=4, phys_dim=2, output_dim=3)
```
Node0: (2, 4)       indices: ("x0", "b0")
Node1: (4, 2, 4)    indices: ("b0", "x1", "b1")
Node2: (4, 2, 3)    indices: ("b1", "x2", "out")
```

### Index Naming
- **Batch**: "s" (sample)
- **Physical**: "x{i}" (input feature at site i)
- **Bonds**: "b{i}" (MPS), "b_mpo_{i}" (MPO), "b_mps_{i}" (MPS in LMPO2)
- **Output**: "out" (single, shared)
- **Primed**: "{index}_prime" (Hessian second derivatives)
- **Tags**: "Node{i}" (site), "NT" (non-trainable)

### Forward Pass
```
y = contract(TN ⊗ inputs, output_inds)
```

### Environment-Based Optimization
```
E = contract(TN \ {node} ⊗ inputs, node_inds ∪ output_inds)
∇_node L = contract(E ⊗ ∇_y L, node_inds)
H_node = contract(E ⊗ ∇²_y L ⊗ E†, node_inds ⊗ node_inds')
```

### Newton Update
```
(H + 2λI) * Δw = -∇L + 2λ * w_old
```

## 🔍 Model Classes

### MPO2
- **Purpose**: Standard MPS with output dimension
- **Structure**: Chain of tensors with bonds
- **Indices**: x{i} (physical), b{i} (bonds), out (output)
- **Use case**: Basic regression/classification

### CMPO2
- **Purpose**: Cross-connected pixel and patch MPS
- **Structure**: Two separate MPS layers
- **Indices**: {i}_pixels, {i}_patches
- **Use case**: Multi-scale feature processing

### LMPO2
- **Purpose**: Dimensionality reduction via MPO + output via MPS
- **Structure**: Two-layer (MPO → MPS)
- **Indices**: {i}_in, {i}_reduced, {i}_mpo, {i}_mps
- **Use case**: Feature reduction before output

### MMPO2
- **Purpose**: Non-trainable causal mask + trainable MPS
- **Structure**: Fixed mask layer + trainable MPS
- **Indices**: {i}_in, {i}_masked, {i}_mps
- **Use case**: Causal/sequential processing

## 📈 Training Algorithms

### Regression (MSE/MAE/Huber)
- Uses direct least squares solver
- Diagonal Hessian approximation
- Fast convergence

### Classification (CrossEntropy)
- Uses Newton-based optimization
- Full Hessian matrix (softmax coupling)
- More iterations needed

## 🛠️ Common Tasks

### Create a Model
```python
from model.standard.MPO2_models import MPO2
model = MPO2(L=3, bond_dim=4, phys_dim=2, output_dim=1)
```

### Prepare Data
```python
from model.utils import create_inputs
inputs = create_inputs(X, y, input_labels=["x"], append_bias=True)
```

### Train
```python
from model.base.NTN import NTN
trainer = NTN(tn=model.tn, output_dims=model.output_dims, 
              input_dims=model.input_dims, loss=loss, data_stream=inputs)
trainer.fit(n_epochs=10, regularize=True, jitter=1e-6)
```

## 📋 Files Analyzed

- ✅ model/standard/MPO2_models.py (441 lines)
- ✅ model/base/NTN.py (1096 lines)
- ✅ model/utils.py (283 lines)
- ✅ model/initialization.py (411 lines)
- ✅ model/losses.py (516 lines)
- ✅ model/builder.py (157 lines)
- ✅ model/base/GTN.py (82 lines)

**Total: 2,986 lines of code analyzed**

## 🎓 Learning Path

1. **Beginner**: Read ANALYSIS_SUMMARY.txt + Example 1
2. **Intermediate**: Read GTN_ARCHITECTURE_ANALYSIS.md sections 1-3 + Examples 1-3
3. **Advanced**: Read all of GTN_ARCHITECTURE_ANALYSIS.md + all examples
4. **Expert**: Study code directly with documentation as reference

## 🔗 Cross-References

- **Tensor shapes**: See GTN_QUICK_REFERENCE.md Section 1 or GTN_ARCHITECTURE_ANALYSIS.md Section 6
- **Forward pass**: See GTN_QUICK_REFERENCE.md Section 2 or GTN_ARCHITECTURE_ANALYSIS.md Section 2.1
- **Initialization**: See GTN_QUICK_REFERENCE.md Section 4 or GTN_ARCHITECTURE_ANALYSIS.md Section 4
- **Examples**: See GTN_DETAILED_EXAMPLES.md
- **Debugging**: See GTN_QUICK_REFERENCE.md Section 11

## 📝 Notes

- All tensor shapes are exact and verified against source code
- Index naming follows consistent conventions throughout
- Mathematical formulations are complete and precise
- Examples are self-contained and runnable
- Documentation covers all 4 MPO2 variants and all loss functions

## ✨ Key Insights

1. **Environment-based optimization** is the core efficiency mechanism
2. **Index naming** is critical for correct tensor contractions
3. **Initialization** significantly affects training stability
4. **Regularization** (jitter) prevents singular matrices
5. **Non-trainable tensors** (NT tag) enable fixed components like masks

---

**Last Updated**: 2024
**Analysis Scope**: Complete GTN architecture
**Code Coverage**: 100% of core training components
