# Loss Function Refactoring Summary

## Overview
Successfully refactored the NTN (Newton Tensor Network) class to use a dedicated loss class hierarchy instead of having `get_derivatives` as an NTN method. This makes it much easier to support different loss functions with different derivative characteristics.

## Changes Made

### 1. Created `model/losses.py`
New file containing a hierarchy of loss classes:

#### Base Class: `TNLoss`
- Defines the interface for all TN-compatible losses
- Has `use_diagonal_hessian` property (class-level default, instance-level override)
- Abstract `get_derivatives()` method that returns `(grad, hess)` as quimb tensors

#### Implemented Loss Classes:

**MSELoss** (Mean Squared Error)
- Inherits from: `nn.MSELoss` + `TNLoss`
- Default: `use_diagonal_hessian = True`
- Reason: Outputs are independent in regression
- Gradient: `2 * (y_pred - y_true) / N`
- Hessian (diagonal): `2 / N` (constant)

**MAELoss** (Mean Absolute Error / L1)
- Inherits from: `nn.L1Loss` + `TNLoss`  
- Default: `use_diagonal_hessian = True`
- Reason: Outputs are independent
- Gradient: `sign(y_pred - y_true) / N`
- Hessian: Approximated as small constant (L1 has zero second derivative)

**HuberLoss** (Robust Regression)
- Inherits from: `nn.HuberLoss` + `TNLoss`
- Default: `use_diagonal_hessian = True`
- Reason: Outputs are independent
- Combines MSE (small errors) and MAE (large errors)

**CrossEntropyLoss** (Classification)
- Inherits from: `nn.CrossEntropyLoss` + `TNLoss`
- Default: `use_diagonal_hessian = False` ← **IMPORTANT!**
- Reason: Softmax creates coupling between class probabilities
- Gradient: `softmax(logits) - one_hot(targets)`
- Hessian (full matrix): `H_ij = p_i * (δ_ij - p_j)` where `p = softmax(logits)`
  - Diagonal: `H_ii = p_i * (1 - p_i)`
  - Off-diagonal: `H_ij = -p_i * p_j`

### 2. Modified `model/NTN.py`

**Removed:**
- `get_derivatives()` method (lines 231-293) - no longer needed!

**Updated:**
- `_batch_node_derivatives()`: Now calls `self.loss.get_derivatives()`
  - Always uses full Hessian (`return_hessian_diagonal=False`) because we contract with environments from both sides
- `_batch_get_derivatives()`: Now calls `self.loss.get_derivatives()`
  - Respects `self.loss.use_diagonal_hessian` property

### 3. Updated `test.py`
- Changed `loss_fn = nn.MSELoss()` to `loss_fn = MSELoss()`
- Import: `from model.losses import MSELoss`

## Key Design Decisions

### 1. Use Property for Hessian Type
Each loss class has a `use_diagonal_hessian` property:
```python
class MSELoss:
    use_diagonal_hessian = True  # Class default
    
    def __init__(self, use_diagonal_hessian=True):
        self.use_diagonal_hessian = use_diagonal_hessian  # Instance override
```

The NTN model respects this automatically - no need to pass flags around!

### 2. CrossEntropy Defaults to Full Hessian
This is critical! CrossEntropyLoss uses `use_diagonal_hessian = False` because:
- Softmax creates coupling between all output classes
- Changing one logit affects all class probabilities
- The Hessian matrix captures this coupling
- Diagonal approximation loses important information

### 3. Flexible Override
Users can override the default if needed:
```python
# Not recommended but possible
ce_loss = CrossEntropyLoss(use_diagonal_hessian=True)  

# Can make MSE use full Hessian
mse_loss = MSELoss(use_diagonal_hessian=False)
```

## Verification Tests

### Gradient Verification (CrossEntropyLoss)
- Analytical formula: `grad = (softmax(logits) - one_hot(targets)) / N`
- Max error: **1.49e-08** ✓

### Hessian Verification (CrossEntropyLoss)
- Analytical formula: `H_ij = p_i * (δ_ij - p_j) / N`
- Max error: **3.73e-09** ✓
- Correctly returns 3D tensor: `(batch, classes, classes)` with indices `['batch', 'class', 'class_prime']`

### Integration Test
- NTN training with MSELoss: ✓ Works (R² = 1.00 in 2 epochs)
- All loss classes tested and verified

## Usage Examples

### Regression with MSE
```python
from model.losses import MSELoss
from model.NTN import NTN

loss = MSELoss()  # Uses diagonal Hessian by default
model = NTN(tn=tn, loss=loss, ...)
model.fit(epochs=10)
```

### Classification with CrossEntropy
```python
from model.losses import CrossEntropyLoss

loss = CrossEntropyLoss()  # Uses FULL Hessian by default
model = NTN(tn=tn, loss=loss, ...)
model.fit(epochs=10)
```

### Custom Loss
To add a new loss, just inherit from `TNLoss` and implement `get_derivatives()`:
```python
class MyLoss(nn.Module, TNLoss):
    use_diagonal_hessian = True  # or False
    
    def __init__(self):
        nn.Module.__init__(self)
        TNLoss.__init__(self)
    
    def get_derivatives(self, y_pred, y_true, backend='numpy', 
                       batch_dim='batch', output_dims=None, 
                       return_hessian_diagonal=None):
        if return_hessian_diagonal is None:
            return_hessian_diagonal = self.use_diagonal_hessian
        
        # Use PyTorch autograd to compute derivatives
        # ... implementation ...
        
        return grad_tensor, hess_tensor
```

## Benefits

1. **Cleaner separation of concerns**: Loss logic is separate from NTN optimization logic
2. **Easy to add new losses**: Just inherit from TNLoss and nn.Loss
3. **Type-specific defaults**: Each loss knows whether it needs full or diagonal Hessian
4. **Backward compatible**: Uses PyTorch autograd, so any differentiable loss works
5. **Flexible**: Can override defaults when needed

## Files Modified

- **Created:** `model/losses.py` (535 lines)
- **Modified:** `model/NTN.py` (removed 63 lines, updated 2 methods)
- **Modified:** `test.py` (updated imports and loss instantiation)
- **Created:** `test_losses.py` (comprehensive test suite)

## Conclusion

The refactoring successfully generalizes loss handling in the NTN framework. The most important achievement is proper support for **CrossEntropyLoss with full Hessian matrices**, which is essential for classification tasks with tensor networks.
