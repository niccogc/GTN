# GTN - Tensor Network Training Framework

## Purpose
Framework for training tensor network models (MPS, MPO2) for machine learning tasks using two approaches:
- **NTN (Newton Tensor Network)**: Second-order optimization using Hessian
- **GTN (Gradient Tensor Network)**: First-order optimization with autograd

## Tech Stack
- Python 3.10+
- PyTorch (autograd, tensors)
- quimb (tensor network operations)
- JAX (backend support)
- scikit-learn (datasets, metrics)

## Core Structure
```
model/
  NTN.py          - Newton-based tensor network training
  GTN.py          - Gradient-based tensor network training  
  MPO2_models.py  - MPO2, LMPO2, MMPO2 model definitions
  MPS.py          - MPS model definitions
  builder.py      - Inputs class for data loading
  losses.py       - Loss functions with derivatives
  utils.py        - Metrics, helpers
  initialization.py - Weight initialization

testing/         - Test scripts for models
testing_typeI/   - Type I ensemble models
experiments/     - Experiment scripts
```

## Key Patterns
- NTN uses `update_tn_node()` for per-node optimization
- GTN uses PyTorch autograd with `forward()` and `backward()`
- Models inherit from NTN/GTN base classes
- `Inputs` class handles batched data loading with quimb tensors
