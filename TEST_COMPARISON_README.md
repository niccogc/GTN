# GTN vs NTN Comparison Test

## Overview
`test_grad_comparison.py` provides a side-by-side comparison of two tensor network training approaches on MNIST classification:

1. **GTN (Gradient Tensor Network)**: First-order gradient descent with PyTorch autograd
2. **NTN (Newton Tensor Network)**: Second-order Newton method with full Hessian

## What It Tests

### Same Architecture
Both models use identical tensor network architecture:
- **Patches MPS**: Processes spatial patches (50 patches)
- **Pixels MPS**: Processes pixel features (17 pixels per patch)
- **Bond dimension**: 6 (for faster testing)
- **Output**: 10 classes (digits 0-9)

### Same Data
- **Train**: 5,000 MNIST samples (subset for faster testing)
- **Test**: 1,000 MNIST samples
- **Preprocessing**: Unfold 28x28 images into 50 patches of 16 pixels each

### Different Optimization
- **GTN**: 
  - Uses `torch.optim.Adam` with learning rate 1e-2
  - Batch size: 100
  - Backpropagation through PyTorch autograd
  - Loss: `nn.CrossEntropyLoss()`

- **NTN**:
  - Uses Newton method with Cholesky decomposition
  - Batch size: 500 (larger batches for stability)
  - Second-order derivatives from `CrossEntropyLoss.get_derivatives()`
  - **Automatically uses FULL Hessian** (not diagonal) because CrossEntropy sets `use_diagonal_hessian=False`

## Key Differences

| Aspect | GTN | NTN |
|--------|-----|-----|
| Order | 1st order (gradients only) | 2nd order (gradients + Hessian) |
| Method | Gradient descent (Adam) | Newton method |
| Derivatives | PyTorch autograd | Custom `get_derivatives()` |
| Hessian | Not used | Full matrix (10x10 per sample) |
| Batch size | 100 | 500 |
| Speed | Faster per epoch | Slower per epoch |
| Convergence | May need more epochs | Fewer epochs typically |

## Why This Test Matters

### 1. **Validates CrossEntropyLoss Implementation**
The NTN successfully trains with CrossEntropyLoss using the **full Hessian matrix**. This proves:
- `get_derivatives()` returns correct gradients and Hessians
- Full Hessian shape `(batch, classes, classes)` is handled correctly
- NTN can optimize classification problems (not just regression)

### 2. **Compares Optimization Methods**
Shows the practical difference between:
- **1st order** (GTN): Fast, works well with good learning rate
- **2nd order** (NTN): More expensive, but uses curvature information

### 3. **Demonstrates Flexibility**
Same TN architecture can be trained with either method:
```python
# GTN approach
model_gtn = Conv(tn=create_tn(), ...)
optimizer = optim.Adam(model_gtn.parameters())

# NTN approach  
model_ntn = NTN(tn=create_tn(), loss=CrossEntropyLoss(), ...)
model_ntn.fit(n_epochs=5)
```

## Expected Behavior

### CrossEntropyLoss in NTN
When NTN trains with CrossEntropyLoss:
1. **Full Hessian is computed** automatically (`use_diagonal_hessian=False`)
2. Shape: `(batch=500, classes=10, classes_prime=10)`
3. Each Hessian is 10x10 matrix capturing class probability coupling
4. NTN contracts Hessian with environments: `H_node = env & H_loss & env_prime`

### Typical Results
- **GTN**: May reach 60-80% accuracy in 5 epochs (with proper tuning)
- **NTN**: May reach similar or better accuracy, but with different convergence pattern
- **Comparison**: NTN can converge faster (fewer epochs) but each epoch is slower

## Running the Test

```bash
python test_grad_comparison.py
```

### Output
1. Progress bars for GTN training
2. NTN Newton sweep messages
3. Epoch-by-epoch comparison table
4. Final accuracy comparison
5. Plot: `gtn_vs_ntn_comparison.png`

### Example Output
```
======================================================================
EPOCH-BY-EPOCH COMPARISON
======================================================================
Epoch    GTN Loss     GTN Acc (%)  NTN Acc (%)  Difference  
----------------------------------------------------------------------
1        2.1234       35.20        38.50        +3.30%
2        1.8456       45.60        52.10        +6.50%
3        1.5234       55.30        61.20        +5.90%
4        1.3012       62.40        68.50        +6.10%
5        1.1456       68.70        73.20        +4.50%
```

## Customization

### Adjust Subset Size
```python
SUBSET_SIZE = 10000  # Use more data
TEST_SIZE = 2000
```

### Adjust Bond Dimension
```python
bond_dim = 8  # Larger model (slower but more expressive)
```

### Adjust Training Parameters
```python
# GTN
optimizer_gtn = optim.Adam(model_gtn.parameters(), lr=5e-3)

# NTN
model_ntn.fit(n_epochs=10, regularize=True, jitter=1e-4)
```

## Interpreting Results

### If NTN is Better
- Newton method benefits from curvature information
- Full Hessian captures class probability coupling
- May indicate the loss landscape has useful 2nd-order structure

### If GTN is Better
- Adam's adaptive learning rate works well
- Newton method may need better regularization
- Hessian computation overhead may not be worth it for this problem

### If Similar
- Both methods are viable
- Choice depends on:
  - Computational resources
  - Training time budget
  - Need for fine-tuning vs rapid prototyping

## Technical Notes

### Memory Usage
- **GTN**: Lower (no Hessian storage)
- **NTN**: Higher (stores full Hessian matrices)
  - Per batch: `batch_size × num_classes × num_classes` floats
  - Example: 500 × 10 × 10 = 50,000 elements per batch

### Debugging
Both models print debug information:
```python
# NTN prints from CrossEntropyLoss
[CrossEntropyLoss.get_derivatives] grad shape: torch.Size([500, 10])
[CrossEntropyLoss.get_derivatives] hess shape: torch.Size([500, 10, 10])
[CrossEntropyLoss.get_derivatives] hess_inds: ['s', 'class_out', 'class_out_prime']
```

This confirms:
- Gradients: (batch=500, classes=10)
- Hessian: (batch=500, classes=10, classes_prime=10) ← Full matrix!

## Conclusion

This test demonstrates that:
1. ✅ NTN successfully trains with CrossEntropyLoss
2. ✅ Full Hessian matrices are computed and used correctly
3. ✅ Both GTN and NTN can optimize the same architecture
4. ✅ The loss refactoring enables flexible, loss-specific Hessian handling

The ability to use full Hessian matrices for CrossEntropyLoss is crucial for classification tasks, as it properly captures the coupling between class probabilities introduced by the softmax function.
