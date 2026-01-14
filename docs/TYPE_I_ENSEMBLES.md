# Type I Ensemble Models

## Overview

Type I ensemble models combine multiple tensor network models with **varying numbers of sites** (L=1, 2, 3, ..., max_sites) into a single ensemble. The key innovation is that all models train together with proper ensemble derivatives.

## Mathematical Foundation

### Ensemble Forward Pass
```
f_ensemble(x) = f_1(x) + f_2(x) + ... + f_n(x)
```

where each `f_i` is a tensor network with `i` sites.

### Ensemble Derivatives

The loss is computed on the ensemble prediction:
```
L = loss(f_ensemble(x), y)
```

For proper training, each model needs derivatives w.r.t. the **ensemble** loss:
```
dL/dθ_i = (dL/df_ensemble) * (df_ensemble/dθ_i)
```

Since `df_ensemble/dθ_i = df_i/dθ_i`, we need `dL/df_ensemble` for each model's update.

### Implementation Strategy

**NTN (Newton-based):**
- Override `forward_from_environment` to add other models' cached outputs
- Cache all model outputs before node updates
- Invalidate cache when a model's nodes are updated

**GTN (Gradient-based):**
- Simply sum outputs in forward pass
- PyTorch autograd handles gradients automatically

## Available Models

### NTN-Based Ensembles

#### MPO2TypeI
Simple MPS ensemble with varying sites.

```python
from model.typeI import MPO2TypeI
from model.losses import CrossEntropyLoss

ensemble = MPO2TypeI(
    max_sites=4,           # Creates models with L=1,2,3,4
    bond_dim=2,
    phys_dim=10,
    output_dim=3,
    loss=CrossEntropyLoss(),
    X_train=X_train,
    y_train=y_train,
    X_val=X_val,
    y_val=y_val,
    batch_size=32,
    init_strength=0.001,
)

scores_train, scores_val = ensemble.fit(
    n_epochs=10,
    regularize=True,
    jitter=0.01,
    eval_metrics=CLASSIFICATION_METRICS,
)
```

**Structure per model (L sites):**
- L MPS nodes with one output dimension
- Total trainable nodes: 1 + 2 + 3 + 4 = 10 (for max_sites=4)

#### LMPO2TypeI
MPO (trainable) for dimensionality reduction + MPS (trainable) for output.

```python
from model.typeI import LMPO2TypeI

ensemble = LMPO2TypeI(
    max_sites=4,
    bond_dim=2,
    phys_dim=10,
    reduced_dim=5,         # Reduction dimension
    output_dim=3,
    loss=loss_fn,
    X_train=X_train,
    y_train=y_train,
    X_val=X_val,
    y_val=y_val,
    batch_size=32,
    init_strength=0.001,
)
```

**Structure per model (L sites):**
- L MPO nodes (trainable): input_dim → reduced_dim
- L MPS nodes (trainable): reduced_dim → output
- Total trainable nodes: 2 + 4 + 6 + 8 = 20 (for max_sites=4)

#### MMPO2TypeI
MPO mask (non-trainable) + MPS (trainable) for output.

```python
from model.typeI import MMPO2TypeI

ensemble = MMPO2TypeI(
    max_sites=4,
    bond_dim=2,
    phys_dim=10,
    output_dim=3,
    loss=loss_fn,
    X_train=X_train,
    y_train=y_train,
    X_val=X_val,
    y_val=y_val,
    batch_size=32,
    init_strength=0.001,
)
```

**Structure per model (L sites):**
- L MPO mask nodes (non-trainable, tagged 'NT'): cumulative sum mask
- L MPS nodes (trainable): masked_input → output
- Total trainable nodes: 1 + 2 + 3 + 4 = 10 (for max_sites=4)

### GTN-Based Ensembles

#### MPO2TypeI_GTN
Simple MPS ensemble using PyTorch autograd.

```python
from model.typeI import MPO2TypeI_GTN
import torch.nn as nn
import torch.optim as optim

model = MPO2TypeI_GTN(
    max_sites=4,
    bond_dim=4,
    phys_dim=10,
    output_dim=3,
    init_strength=0.1,
)

optimizer = optim.Adam(model.parameters(), lr=0.01)
loss_fn = nn.CrossEntropyLoss()

for epoch in range(100):
    optimizer.zero_grad()
    y_pred = model(X_train)
    loss = loss_fn(y_pred, y_train)
    loss.backward()
    optimizer.step()
```

#### LMPO2TypeI_GTN
MPO reduction + MPS ensemble using PyTorch autograd.

```python
from model.typeI import LMPO2TypeI_GTN

model = LMPO2TypeI_GTN(
    max_sites=4,
    bond_dim=4,
    phys_dim=10,
    reduced_dim=5,
    output_dim=3,
    init_strength=0.1,
)
```

#### MMPO2TypeI_GTN
MPO mask + MPS ensemble using PyTorch autograd.

```python
from model.typeI import MMPO2TypeI_GTN

model = MMPO2TypeI_GTN(
    max_sites=4,
    bond_dim=4,
    phys_dim=10,
    output_dim=3,
    init_strength=0.1,
)
```

## Key Differences: NTN vs GTN

| Aspect | NTN TypeI | GTN TypeI |
|--------|-----------|-----------|
| **Training** | Newton's method (2nd order) | Gradient descent (1st order) |
| **Derivatives** | Manual Hessian computation | PyTorch autograd |
| **Ensemble** | Custom `forward_from_environment` | Simple output summation |
| **Caching** | Required for efficiency | Not needed |
| **Data** | Separate loaders per model | Single input tensor |
| **Speed** | Slower per epoch | Faster per epoch |
| **Convergence** | Fewer epochs needed | More epochs needed |

## Performance Comparison (Iris Dataset)

| Model | Method | Train Acc | Val Acc | Nodes |
|-------|--------|-----------|---------|-------|
| MPO2TypeI | NTN | 100% | 86.7% | 10 |
| LMPO2TypeI | NTN | 100% | 91.1% | 20 |
| MMPO2TypeI | NTN | 100% | 91.1% | 10 |
| MPO2TypeI_GTN | GTN | 99.1% | 93.3% | 396 params |
| LMPO2TypeI_GTN | GTN | 99.1% | 93.3% | 1173 params |
| MMPO2TypeI_GTN | GTN | 100% | 88.9% | 396 params |

## When to Use Type I Ensembles

**Advantages:**
- Combines models of different complexities
- Can capture both simple and complex patterns
- Often better generalization than single models
- Automatic model selection through training

**Use when:**
- Uncertain about optimal model size
- Want robustness across different scales
- Have enough data to train multiple models
- Computational cost is acceptable

**Avoid when:**
- Very limited training data
- Need maximum speed/efficiency
- Single model size is clearly optimal
- Interpretability is critical

## Implementation Details

### NTN TypeI Architecture

```python
class NTN_TypeI(NTN):
    """Custom NTN that adds other models' outputs in forward_from_environment."""
    
    def forward_from_environment(self, env, node_tag=None, node_tensor=None, sum_over_batch=False):
        # Compute this model's prediction
        y_pred_self = super().forward_from_environment(env, node_tag, node_tensor, sum_over_batch)
        
        # Add cached predictions from other models
        y_pred_others = self.get_others_cached_output_fn(self.ntn_index, self._batch_idx)
        
        if y_pred_others is not None:
            return y_pred_self + y_pred_others
        return y_pred_self
```

### Caching Strategy

1. **Before epoch**: Cache forward outputs for all models
2. **During training**: 
   - Update nodes of model i
   - Invalidate cache for model i
   - Continue to next model
3. **For evaluation**: Cache outputs on train/val/test splits separately

### GTN TypeI Architecture

```python
class MPO2TypeI_GTN(nn.Module):
    """Simple ensemble - PyTorch handles gradients automatically."""
    
    def forward(self, x):
        total = None
        for model in self.models:
            y = model(x)
            if total is None:
                total = y
            else:
                total = total + y
        return total
```

## Examples

See `testing_typeI/` directory for complete examples:
- `test_mpo2_typeI_iris.py` - MPO2TypeI on Iris
- `test_lmpo2_typeI_iris.py` - LMPO2TypeI on Iris
- `test_mmpo2_typeI_iris.py` - MMPO2TypeI on Iris
- `test_gtn_typeI_iris.py` - All GTN TypeI models on Iris

## References

- Type I ensembles combine models of varying complexity
- Proper ensemble derivatives ensure all models contribute to learning
- Caching strategy enables efficient NTN training
- GTN implementation is simpler due to autograd
