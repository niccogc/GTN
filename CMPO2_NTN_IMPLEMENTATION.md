# CMPO2_NTN Implementation Summary

## Overview

Created `CMPO2_NTN` class in `model/MPS.py` - a specialized NTN subclass for cascaded matrix product operator structures with environment caching support.

## Files Created/Modified

### 1. `model/MPS.py` (NEW)
- **CMPO2_NTN class**: Inherits from NTN
- **from_tensors()**: Class method to construct from two MPS layers
- **_batch_environment()**: Overrideable method for cached environment computation
- **Status**: ✅ Basic implementation working

### 2. `batch_moving_environment.py` (NEW)
- **BatchMovingEnvironment class**: Extends quimb's MovingEnvironment
- Handles batch dimensions during environment caching
- Preserves batch indices during contraction
- **Status**: ✅ Fully functional and tested

### 3. `BATCH_MOVING_ENVIRONMENT.md` (NEW)
- Complete documentation of BatchMovingEnvironment
- Usage examples and performance notes
- **Status**: ✅ Complete

### 4. `test_cmpo2_basic.py` (NEW)
- Basic functionality test for CMPO2_NTN
- Tests model creation, forward pass, training
- **Status**: ✅ Passing

## CMPO2_NTN Architecture

### Structure
```
CMPO2 (Cascaded Matrix Product Operator - 2 layers)
├── MPS1 (e.g., Pixel MPS)
│   ├── Site 0: (pixels, bond)
│   ├── Site 1: (bond, pixels, bond, output)
│   └── Site 2: (bond, pixels)
└── MPS2 (e.g., Patch MPS)
    ├── Site 0: (r1, patches, bond)
    ├── Site 1: (bond, patches, bond)
    └── Site 2: (bond, patches, r1)
```

### Input Format
For 3D data (samples, patches, pixels):
```python
input_labels_ntn = [
    [0, ("0_patches", "0_pixels")],  # Site 0
    [0, ("1_patches", "1_pixels")],  # Site 1
    [0, ("2_patches", "2_pixels")]   # Site 2
]
```

## Current Implementation Status

### ✅ Working Features
1. **Model Creation**
   - `from_tensors()` class method
   - Combines two MPS layers into single TensorNetwork
   - Proper initialization with cache_environments flag

2. **Training**
   - Full Newton optimization working
   - Gradient and Hessian computation
   - Sequential node updates in sweep

3. **Basic Functionality**
   - Forward pass
   - Loss computation
   - Evaluation metrics
   - Multi-epoch training

### 🚧 TODO: Environment Caching

Currently `_batch_environment()` falls back to parent implementation. To implement caching:

```python
def _batch_environment(self, inputs, tn, target_tag, sum_over_batch, sum_over_output):
    if not self.cache_environments:
        return super()._batch_environment(...)
    
    # 1. Create cache key from batch
    cache_key = id(inputs)
    
    # 2. Get or create BatchMovingEnvironment
    if cache_key not in self._env_cache:
        full_tn = tn & inputs
        self._env_cache[cache_key] = BatchMovingEnvironment(
            full_tn, 
            begin='left', 
            bsz=1, 
            batch_inds=[self.batch_dim]
        )
    
    env_obj = self._env_cache[cache_key]
    
    # 3. Map target_tag to position
    site_idx = self._get_site_index(target_tag)
    env_obj.move_to(site_idx)
    
    # 4. Get environment and delete target
    current_env = env_obj()
    env_with_hole = current_env.copy()
    env_with_hole.delete(target_tag)
    
    # 5. Contract with proper indices
    final_inds = self._get_final_indices(env_with_hole, sum_over_batch, sum_over_output)
    env_tensor = env_with_hole.contract(output_inds=final_inds)
    
    return env_tensor
```

### Key Challenges for Caching

1. **Site Index Mapping**
   - Need to map node tags (e.g., "0_Pi", "1_Pa") to sequential positions
   - Different orderings for different sweeps

2. **Cache Management**
   - When to create/clear cache
   - Memory vs speed tradeoff
   - Per-batch vs per-epoch caching

3. **Index Handling**
   - Batch dimension preservation
   - Output dimension handling
   - Bond indices from target node

## Test Results

### test_cmpo2_basic.py
```
Setup: 50 samples, batch_size=10, patches=5, pixels=4, bond_dim=2
Created: 6 trainable nodes (3 pixel MPS + 3 patch MPS)
Training: 1 epoch
Results:
  Init MSE: 1.051
  Final MSE: 0.292
  R²: 0.722
Status: ✅ PASS
```

## Performance Comparison (Planned)

Once caching is implemented, compare:

| Method | Time per Epoch | Memory | Notes |
|--------|---------------|---------|-------|
| Without caching | Baseline | Baseline | Current implementation |
| With BatchMovingEnvironment | ~50-70% faster* | +10-20%* | Reuses contractions |

*Estimated based on number of repeated contractions in sweep

## Next Steps

1. **Implement site index mapping**
   - Create `_get_site_index(tag)` method
   - Handle different sweep orderings

2. **Implement full caching in `_batch_environment`**
   - Create/manage BatchMovingEnvironment cache
   - Handle index management properly
   - Test correctness vs non-cached version

3. **Add cache management**
   - Clear cache between epochs if needed
   - Handle memory limits
   - Add cache statistics tracking

4. **Performance testing**
   - Benchmark cached vs non-cached on MNIST
   - Measure memory usage
   - Profile to find bottlenecks

5. **Extended testing**
   - Test on `test_cmpo2_mnist.py`
   - Compare results with/without caching
   - Verify correctness on larger datasets

## Usage Example

```python
from model.MPS import CMPO2_NTN
from model.builder import Inputs
from model.losses import CrossEntropyLoss

# Create MPS tensors
pixels_mps = [...]  # 3 tensors
patches_mps = [...]  # 3 tensors

# Setup data
loader = Inputs(
    inputs=[data],  # (N, patches, pixels)
    outputs=[labels],
    outputs_labels=["class_out"],
    input_labels=[
        [0, ("0_patches", "0_pixels")],
        [0, ("1_patches", "1_pixels")],
        [0, ("2_patches", "2_pixels")]
    ],
    batch_dim="s",
    batch_size=100
)

# Create model
model = CMPO2_NTN.from_tensors(
    mps1_tensors=pixels_mps,
    mps2_tensors=patches_mps,
    output_dims=["class_out"],
    input_dims=["0", "1", "2"],
    loss=CrossEntropyLoss(),
    data_stream=loader,
    cache_environments=True  # Enable caching
)

# Train
model.fit(n_epochs=10, regularize=True, jitter=1e-4, verbose=True)
```

## References

- `NTN._batch_environment()`: Base implementation in model/NTN.py:384-409
- `BatchMovingEnvironment`: Implementation in batch_moving_environment.py
- `test_cmpo2_basic.py`: Basic functionality test
- `test_grad_comparison.py`: Example of NTN usage with MNIST
