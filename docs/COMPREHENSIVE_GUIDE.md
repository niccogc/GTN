# Tensor Network Training: Comprehensive Guide

**Version:** 0.1.0  
**Last Updated:** 2025-01-12

A complete framework for training tensor network models using Newton-based (NTN) and Gradient-based (GTN) optimization methods.

---

## Table of Contents

1. [Introduction](#introduction)
2. [Installation](#installation)
3. [Quick Start](#quick-start)
4. [Training Methods](#training-methods)
5. [Model Architectures](#model-architectures)
6. [Datasets](#datasets)
7. [Grid Search System](#grid-search-system)
8. [Experiment Tracking](#experiment-tracking)
9. [Complete Parameter Reference](#complete-parameter-reference)
10. [Best Practices & Tuning](#best-practices--tuning)
11. [Troubleshooting](#troubleshooting)
12. [API Reference](#api-reference)

---

## Introduction

### What is This Framework?

This framework provides production-ready tools for training tensor network models on machine learning tasks. It supports two complementary training approaches:

**NTN (Newton-based Tensor Network Training)**
- Uses Newton-like optimization with second-order information
- Solves linear systems at each training step
- Ridge regularization (jitter) for numerical stability
- Fast convergence (10-50 epochs typical)
- Lower memory footprint

**GTN (Gradient-based Tensor Network Training)**
- Uses first-order gradient descent via PyTorch autograd
- Standard backpropagation with multiple optimizers
- Flexible loss functions (MSE, MAE, Huber, CrossEntropy)
- Slower convergence (50-200 epochs typical)
- Better integration with PyTorch ecosystem

### Supported Models

**MPO2:** Matrix Product Operator with output dimension
- Simplest model, fewest parameters
- Good for general-purpose tasks

**LMPO2:** Linear MPO with dimensionality reduction
- Includes linear projection layer
- Useful for high-dimensional inputs

**MMPO2:** Masked MPO with cumulative mask
- Attention-like masking mechanism
- Useful for sequential data

### Key Features

- **21 UCI datasets** (11 regression, 10 classification) with automatic loading
- **Automated grid search** with Cartesian product expansion
- **Multi-seed experiments** with statistical analysis
- **Full experiment tracking** (AIM, file-based, or both)
- **Automatic resumption** of interrupted experiments
- **Production-ready scripts** with comprehensive error handling

---

## Installation

### Option 1: Nix Flake (Recommended)

The Nix flake provides all dependencies managed reproducibly:

```bash
git clone <repository-url>
cd GTN
nix develop
```

**Provided by Nix:**
- Python 3.12
- PyTorch, torchvision
- Quimb (tensor network library)
- JAX
- SciPy, NumPy, Pandas
- Matplotlib
- Scikit-learn
- Pytest
- UCI ML Repository package
- UV package manager

**Installed via UV in venv:**
- AIM (experiment tracking)
- Additional pure-Python dependencies

### Option 2: UV (Standalone)

```bash
uv venv
source .venv/bin/activate
uv pip install -e .
```

All dependencies listed in `pyproject.toml`:
- Core: torch, quimb, jax, numpy, pandas, scipy
- ML: scikit-learn
- Tensor networks: cotengra, autoray, opt-einsum, networkx
- Optimization: optuna, kahypar
- Tracking: aim
- Data: ucimlrepo
- Utils: matplotlib, tqdm

### Option 3: pip

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

### Verify Installation

```bash
python -c "import torch, quimb, model, experiments; print('✓ All imports successful')"
python -c "from experiments.dataset_loader import load_dataset; d, i = load_dataset('iris'); print(f'✓ Loaded {i[\"name\"]} dataset')"
```

---

## Quick Start

### 30-Second Test (NTN)

Train MPO2 on iris dataset:

```bash
cd experiments
python train_mpo2.py \
    --dataset iris \
    --bond-dim 6 \
    --jitter-start 5.0 \
    --n-epochs 20 \
    --output-dir ../results/quick_test
```

Expected output: ~87% test accuracy in <10 seconds

### 30-Second Test (GTN)

```bash
cd experiments
python run_grid_search_gtn.py \
    --config configs/iris_gtn_minimal.json \
    --output-dir ../results/iris_gtn_test \
    --tracker file \
    --verbose
```

Expected output: ~87% test accuracy in <5 seconds

### Grid Search (NTN)

Run parameter sweep on iris:

```bash
cd experiments
python run_grid_search.py \
    --config configs/iris_minimal_test.json \
    --verbose
```

This runs 8 configurations × 2 seeds = 16 experiments in ~30 seconds.

### Grid Search (GTN)

```bash
cd experiments
python run_grid_search_gtn.py \
    --config configs/iris_gtn_sweep.json \
    --output-dir ../results/iris_gtn_sweep \
    --tracker file
```

---

## Training Methods

### NTN: Newton-based Training

**Training Loop:**
1. For each tensor in the network:
   - Compute gradient and Hessian
   - Add ridge penalty to diagonal: `H + λI`
   - Solve linear system for weight update
   - Update tensor weights
2. Repeat for all tensors (one sweep)
3. Decay jitter: `λ ← λ × decay`
4. Repeat for n_epochs

**Key Parameters:**

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `jitter_start` | 0.001 (reg), 5.0 (class) | 1e-6 to 10.0 | Initial ridge penalty |
| `jitter_decay` | 0.95 | 0.5 to 0.99 | Multiplicative decay per epoch |
| `jitter_min` | 0.001 | 1e-8 to 1e-3 | Minimum jitter value |
| `adaptive_jitter` | true | true/false | Auto-adjust for ill-conditioned matrices |
| `n_epochs` | 10-50 | 5 to 100 | Number of training epochs |

**Example NTN Config:**
```json
{
  "experiment_name": "iris_ntn",
  "dataset": "iris",
  "task": "classification",
  
  "parameter_grid": {
    "model": ["MPO2"],
    "bond_dim": [6, 8],
    "jitter_start": [1.0, 5.0, 10.0]
  },
  
  "fixed_params": {
    "L": 3,
    "output_site": 1,
    "init_strength": 0.001,
    "n_epochs": 20,
    "batch_size": 100,
    "jitter_decay": 0.95,
    "jitter_min": 0.001,
    "adaptive_jitter": true,
    "patience": 5,
    "min_delta": 0.001,
    "train_selection": true,
    "seeds": [0, 1, 2, 3, 4]
  }
}
```

### GTN: Gradient-based Training

**Training Loop:**
1. Wrap tensor network in `nn.Module`
2. For each mini-batch:
   - Forward pass: contract tensor network
   - Compute loss (MSE, CrossEntropy, MAE, Huber)
   - Backward pass: compute gradients via autograd
   - Update with optimizer (Adam, AdamW, SGD)
3. Validate after each epoch
4. Repeat for n_epochs

**Key Parameters:**

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `lr` | 0.001 | 1e-5 to 0.1 | Learning rate |
| `weight_decay` | 0.01 | 0.0 to 1.0 | L2 regularization |
| `optimizer` | adam | adam/adamw/sgd | Optimizer type |
| `loss_fn` | auto | mse/mae/huber/cross_entropy | Loss function |
| `n_epochs` | 100 | 30 to 500 | Number of training epochs |
| `batch_size` | 32 | 16 to 256 | Mini-batch size |

**When to Use GTN:**
- Custom loss functions needed
- Integration with other PyTorch models
- Transfer learning scenarios
- More flexible optimization required
- Willing to tune learning rate carefully

**Example GTN Config:**
```json
{
  "experiment_name": "iris_gtn",
  "dataset": "iris",
  "task": "classification",
  
  "parameter_grid": {
    "model": ["MPO2"],
    "bond_dim": [6, 8],
    "lr": [0.001, 0.01],
    "weight_decay": [0.0, 0.01, 0.1],
    "optimizer": ["adam", "adamw"],
    "loss_fn": ["cross_entropy"]
  },
  
  "fixed_params": {
    "L": 3,
    "output_site": 1,
    "init_strength": 0.001,
    "n_epochs": 100,
    "batch_size": 32,
    "seeds": [0, 1, 2]
  }
}
```

### Comparison Table

| Aspect | NTN (Newton-based) | GTN (Gradient-based) |
|--------|-------------------|---------------------|
| **Optimization** | Second-order (Newton-like) | First-order (gradient descent) |
| **Speed per epoch** | Fast (analytical solve) | Slower (batched gradients) |
| **Total epochs needed** | 10-50 | 50-200 |
| **Memory usage** | Lower (no autograd) | Higher (autograd graph) |
| **Numerical stability** | Very stable | Can be unstable |
| **Loss function** | Fixed per task | Customizable |
| **Integration** | Standalone | PyTorch ecosystem |
| **Best for** | Quick experiments | Custom objectives |
| **Typical accuracy** | 85-90% (iris) | 85-90% (iris) |

---

## Model Architectures

### MPO2: Matrix Product Operator

**Structure:**

Linear chain of tensors with one designated output tensor.

```
Input₁ --- Tensor₁ --- Tensor₂ --- ... --- TensorL --- Output
           |           |                    |
         Input₂      Input₃              InputL
```

**Parameters:**

```python
MPO2(
    L=3,                    # Number of sites (2-10)
    bond_dim=8,             # Bond dimension (4-20)
    phys_dim=4,             # Physical dimension (auto from data)
    output_dim=3,           # Output dimension (1 for reg, n_classes for class)
    output_site=1,          # Which site carries output (0 to L-1)
    init_strength=0.001,    # Initialization scale
    use_tn_normalization=True  # Frobenius normalization
)
```

**Computational Complexity:**
- Parameters: O(L × phys_dim × bond_dim²)
- Forward pass: O(L × phys_dim × bond_dim³)

**Use Cases:**
- General-purpose tensor network model
- Good starting point for all tasks
- Efficient for low-to-medium dimensional data

**Example:**
```python
from model.MPO2_models import MPO2

model = MPO2(
    L=3,
    bond_dim=8,
    phys_dim=4,
    output_dim=3,
    output_site=1,
    init_strength=0.001
)
```

### LMPO2: Linear MPO with Reduction

**Structure:**

MPO with an additional linear dimensionality reduction layer before the tensor network.

```
Input (high-dim) → Linear Reduction → MPO2 → Output
```

**Additional Parameters:**

```python
LMPO2(
    L=3,
    bond_dim=8,
    phys_dim=100,           # Original input dimension
    output_dim=10,
    rank=5,                 # Reduced dimension
    reduction_factor=0.5,   # Alternative to rank (not used if rank specified)
    output_site=1,
    init_strength=0.001
)
```

**How Reduction Works:**
- If `rank` provided: reduced_dim = rank
- If `reduction_factor` provided: reduced_dim = int(phys_dim × reduction_factor)
- Linear projection: R^phys_dim → R^reduced_dim
- Then feeds into MPO2 with reduced_dim as phys_dim

**Computational Complexity:**
- Reduction: O(phys_dim × reduced_dim)
- MPO: O(L × reduced_dim × bond_dim³)
- Total often lower than direct MPO2 on high-dim data

**Use Cases:**
- High-dimensional inputs (phys_dim > 50)
- Feature compression beneficial
- Noisy or redundant features

**Example:**
```python
from model.MPO2_models import LMPO2

model = LMPO2(
    L=3,
    bond_dim=8,
    phys_dim=100,  # High-dimensional input
    output_dim=10,
    rank=5,        # Reduce to 5 dimensions
    output_site=1
)
```

### MMPO2: Masked MPO

**Structure:**

MPO with a cumulative masking mechanism that creates attention-like behavior.

```
Input₁ --[Mask₁]--> Tensor₁ --[Mask₂]--> Tensor₂ --> ... --> Output
```

**Additional Parameters:**

```python
MMPO2(
    L=4,
    bond_dim=6,
    phys_dim=20,
    output_dim=5,
    rank=5,                 # Mask complexity
    output_site=1,
    init_strength=0.001
)
```

**How Masking Works:**
- Creates cumulative mask tensors between MPO tensors
- Mask complexity controlled by `rank` parameter
- Allows network to selectively attend to different features

**Use Cases:**
- Sequential or temporal data
- When feature interactions are important
- Attention-like mechanisms needed

**Example:**
```python
from model.MPO2_models import MMPO2

model = MMPO2(
    L=4,
    bond_dim=6,
    phys_dim=20,
    output_dim=5,
    rank=5
)
```

### Model Comparison Table

| Aspect | MPO2 | LMPO2 | MMPO2 |
|--------|------|-------|-------|
| **Parameters** | L × phys_dim × bond_dim² | + phys_dim × rank | + L × rank² |
| **Complexity** | Low | Medium | High |
| **Best for** | General tasks | High-dim input | Sequential data |
| **Feature selection** | None | Implicit (reduction) | Attention-like |
| **Memory** | Lowest | Medium | Highest |
| **Training speed** | Fastest | Medium | Slowest |

---

## Datasets

### UCI ML Repository Integration

All datasets are loaded automatically from the UCI ML Repository using the `ucimlrepo` package. No manual download needed.

### Available Datasets (21 total)

#### Classification (10 datasets)

| Dataset | Samples | Features | Classes | Description |
|---------|---------|----------|---------|-------------|
| `iris` | 150 | 4 | 3 | Iris flower species |
| `wine` | 178 | 13 | 3 | Wine recognition |
| `breast_cancer` | 569 | 30 | 2 | Breast cancer Wisconsin |
| `glass` | 214 | 9 | 6 | Glass identification |
| `ecoli` | 336 | 7 | 8 | E. coli protein localization |
| `yeast` | 1484 | 8 | 10 | Yeast protein localization |
| `car_evaluation` | 1728 | 6 | 4 | Car evaluation |
| `mushroom` | 8124 | 22 | 2 | Mushroom edibility |
| `tic_tac_toe` | 958 | 9 | 2 | Tic-tac-toe endgames |
| `spam` | 4601 | 57 | 2 | Spam email detection |

#### Regression (11 datasets)

| Dataset | Samples | Features | Description |
|---------|---------|----------|-------------|
| `california_housing` | 20640 | 8 | California housing prices |
| `boston_housing` | 506 | 13 | Boston housing prices |
| `diabetes` | 442 | 10 | Diabetes progression |
| `abalone` | 4177 | 8 | Abalone age prediction |
| `airfoil` | 1503 | 5 | Airfoil self-noise |
| `concrete` | 1030 | 8 | Concrete compressive strength |
| `energy_cooling` | 768 | 8 | Energy efficiency (cooling) |
| `energy_heating` | 768 | 8 | Energy efficiency (heating) |
| `power_plant` | 9568 | 4 | Power plant energy output |
| `yacht` | 308 | 6 | Yacht hydrodynamics |
| `wine_quality` | 4898 | 11 | Wine quality rating |

### Data Loading

**Automatic preprocessing:**
1. Missing values dropped
2. Categorical variables one-hot encoded (max 50 features)
3. Numeric features standardized (zero mean, unit variance)
4. Train/Val/Test split: 70%/15%/15% (seed=42)

**Output format:**
- Regression: targets shape `(n, 1)`
- Classification: one-hot targets shape `(n, n_classes)`
- All tensors are torch.float64

**Usage:**

```python
from experiments.dataset_loader import load_dataset

data, dataset_info = load_dataset('iris')

print(f"Dataset: {dataset_info['name']}")
print(f"Task: {dataset_info['task']}")
print(f"Train: {dataset_info['n_train']} samples")
print(f"Val: {dataset_info['n_val']} samples")
print(f"Test: {dataset_info['n_test']} samples")
print(f"Features: {dataset_info['n_features']}")
print(f"Classes: {dataset_info.get('n_classes', 'N/A')}")

X_train, y_train = data['X_train'], data['y_train']
X_val, y_val = data['X_val'], data['y_val']
X_test, y_test = data['X_test'], data['y_test']
```

**Data dict structure:**
```python
{
    'X_train': torch.Tensor (n_train, n_features),
    'y_train': torch.Tensor (n_train, output_dim),
    'X_val': torch.Tensor (n_val, n_features),
    'y_val': torch.Tensor (n_val, output_dim),
    'X_test': torch.Tensor (n_test, n_features),
    'y_test': torch.Tensor (n_test, output_dim)
}
```

**Info dict structure:**
```python
{
    'name': str,
    'task': 'regression' or 'classification',
    'n_features': int,
    'n_train': int,
    'n_val': int,
    'n_test': int,
    'n_classes': int (classification only)
}
```

---

## Grid Search System

### Overview

The grid search system automates hyperparameter tuning by:
1. Defining parameter grids in JSON configs
2. Expanding Cartesian product of all combinations
3. Running each combination with multiple seeds
4. Tracking all experiments automatically
5. Generating summary statistics

### Configuration File Structure

```json
{
  "experiment_name": "iris_grid_search",
  "dataset": "iris",
  "task": "classification",
  
  "parameter_grid": {
    "model": ["MPO2", "LMPO2"],
    "L": [2, 3, 4],
    "bond_dim": [4, 6, 8],
    "jitter_start": [1.0, 5.0, 10.0]
  },
  
  "fixed_params": {
    "output_site": 1,
    "init_strength": 0.001,
    "batch_size": 100,
    "n_epochs": 20,
    "jitter_decay": 0.95,
    "jitter_min": 0.001,
    "adaptive_jitter": true,
    "patience": 5,
    "min_delta": 0.001,
    "train_selection": true,
    "seeds": [0, 1, 2, 3, 4],
    "verbose": false
  },
  
  "tracker": {
    "backend": "file",
    "tracker_dir": "experiment_logs",
    "aim_repo": null
  },
  
  "output": {
    "results_dir": "results/iris_grid_search",
    "save_models": false,
    "save_individual_runs": true
  }
}
```

### How It Works

**1. Grid Expansion:**

The `parameter_grid` creates a Cartesian product:

```
2 models × 3 L values × 3 bond_dims × 3 jitters = 54 combinations
54 combinations × 5 seeds = 270 total experiments
```

**2. Run Identification:**

Each run gets a unique ID:
```
Format: {model}-L{L}-d{bond_dim}-jit{jitter}-seed{seed}
Example: MPO2-L3-d6-jit5-seed0
```

**3. Experiment Grouping:**

All runs for the same dataset are grouped under one experiment name (the dataset name).

**4. Automatic Resumption:**

The system checks for existing result files and skips completed runs:
- Checks if `{results_dir}/{run_id}.json` exists
- Verifies `"success"` field is present
- Skips if already completed

### Running Grid Search

**NTN (Newton-based):**
```bash
python experiments/run_grid_search.py \
    --config experiments/configs/iris_minimal_test.json \
    --verbose
```

**GTN (Gradient-based):**
```bash
python experiments/run_grid_search_gtn.py \
    --config experiments/configs/iris_gtn_sweep.json \
    --output-dir results/iris_gtn_sweep \
    --tracker file \
    --verbose
```

**Command-line options:**
```
--config CONFIG        Path to JSON config file (required)
--verbose              Print training progress for each run
--aim-repo URL         AIM repository URL (for NTN)
--output-dir DIR       Output directory (for GTN)
--tracker BACKEND      Tracking backend: file/aim/both (for GTN)
```

### Output Structure

```
results/iris_grid_search/
├── summary.json                           # Aggregated results
├── MPO2-L2-d4-jit1-seed0.json            # Individual run
├── MPO2-L2-d4-jit1-seed1.json
├── MPO2-L2-d4-jit1-seed2.json
├── MPO2-L2-d4-jit5-seed0.json
└── ...

experiment_logs/
├── iris_MPO2_L2_d4_j1_s0.json            # Detailed tracking
├── iris_MPO2_L2_d4_j1_s1.json
└── ...
```

**summary.json structure:**
```json
{
  "config": {...},
  "dataset_info": {...},
  "total_runs": 270,
  "completed": 268,
  "skipped": 0,
  "failed": 2,
  "elapsed_time": 456.7,
  "top_configurations": [
    {
      "run_id": "MPO2-L3-d6-jit5-seed1",
      "seed": 1,
      "model": "MPO2",
      "dataset": "iris",
      "task": "classification",
      "params": {...},
      "train_quality": 0.95,
      "val_quality": 0.91,
      "test_quality": 0.87,
      "success": true
    },
    ...
  ]
}
```

### Grid Search Examples

**Small test (3 minutes):**
```json
{
  "parameter_grid": {
    "model": ["MPO2"],
    "bond_dim": [4, 6, 8]
  },
  "fixed_params": {
    "L": 3,
    "jitter_start": 1.0,
    "n_epochs": 10,
    "seeds": [0, 1, 2]
  }
}
```
Total runs: 1 × 3 × 3 = 9 experiments

**Medium search (30 minutes):**
```json
{
  "parameter_grid": {
    "model": ["MPO2", "LMPO2"],
    "L": [2, 3, 4],
    "bond_dim": [4, 6, 8],
    "jitter_start": [0.01, 0.1, 1.0]
  },
  "fixed_params": {
    "n_epochs": 20,
    "seeds": [0, 1, 2, 3, 4]
  }
}
```
Total runs: 2 × 3 × 3 × 3 × 5 = 270 experiments

**Large search (4-8 hours):**
```json
{
  "parameter_grid": {
    "model": ["MPO2", "LMPO2", "MMPO2"],
    "L": [2, 3, 4, 5],
    "bond_dim": [4, 6, 8, 10],
    "jitter_start": [0.001, 0.01, 0.1, 1.0, 10.0]
  },
  "fixed_params": {
    "n_epochs": 50,
    "seeds": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
  }
}
```
Total runs: 3 × 4 × 4 × 5 × 10 = 2400 experiments

---

## Experiment Tracking

### Tracking Backends

The framework supports four tracking backends:

**1. File (default)** - JSON files with epoch-by-epoch metrics
**2. AIM** - Interactive web UI with real-time visualization
**3. Both** - Simultaneously log to file and AIM
**4. None** - No tracking (for quick tests)

### File-based Tracking

**Configuration:**
```json
{
  "tracker": {
    "backend": "file",
    "tracker_dir": "experiment_logs"
  }
}
```

**Output format:**
```json
{
  "experiment_name": "iris_MPO2_L3_d6_j5_s0",
  "config": {
    "dataset": "iris",
    "model": "MPO2",
    "L": 3,
    "bond_dim": 6,
    "jitter_start": 5.0,
    ...
  },
  "hparams": {
    "seed": 0,
    "model": "MPO2",
    "L": 3,
    "bond_dim": 6,
    ...
  },
  "metrics_log": [
    {"step": -1, "train_loss": 6.64, "train_quality": -3.88, "val_loss": 6.89, "val_quality": -4.12},
    {"step": 0, "train_loss": 0.27, "train_quality": 0.80, "val_loss": 0.31, "val_quality": 0.75, "reg_loss": 0.86, "jitter": 5.0},
    {"step": 1, "train_loss": 0.26, "train_quality": 0.81, "val_loss": 0.30, "val_quality": 0.76, "reg_loss": 0.80, "jitter": 4.75},
    ...
  ],
  "summary": {
    "seed": 0,
    "train_loss": 0.25,
    "train_quality": 0.82,
    "val_loss": 0.29,
    "val_quality": 0.77,
    "test_loss": 0.30,
    "test_quality": 0.87,
    "best_epoch": 15,
    "success": true
  }
}
```

**Viewing results:**
```bash
cat experiment_logs/iris_MPO2_L3_d6_j5_s0.json

python -c "
import json
with open('experiment_logs/iris_MPO2_L3_d6_j5_s0.json') as f:
    data = json.load(f)
print(f\"Test Quality: {data['summary']['test_quality']:.4f}\")
"
```

### AIM Tracking

**Configuration:**
```json
{
  "tracker": {
    "backend": "aim",
    "tracker_dir": "experiment_logs",
    "aim_repo": "aim://192.168.5.5:5800"
  }
}
```

**AIM repository options:**
- `null` or `.aim`: Local AIM repository
- `aim://host:port`: Remote AIM server
- `aim://192.168.5.5:5800`: VPN access
- `aim://aimtracking.kosmon.org:443`: Non-VPN access (requires auth)

**AIM Features:**
- Real-time metric visualization
- Compare runs side-by-side
- Filter by hyperparameters
- Export results
- Search experiment history

**Authentication (non-VPN):**
```bash
export CF_ACCESS_CLIENT_ID="your_client_id"
export CF_ACCESS_CLIENT_SECRET="your_client_secret"

python experiments/run_grid_search.py \
    --config config.json \
    --aim-repo aim://aimtracking.kosmon.org:443
```

### Both Trackers

Track to both file and AIM simultaneously:

```json
{
  "tracker": {
    "backend": "both",
    "tracker_dir": "experiment_logs",
    "aim_repo": "aim://192.168.5.5:5800"
  }
}
```

This provides:
- Local JSON files for offline analysis
- AIM UI for interactive exploration
- Backup in case one fails

### No Tracking

Disable tracking for quick tests:

```json
{
  "tracker": {
    "backend": "none"
  }
}
```

### Tracked Metrics

**Per Epoch (NTN):**
- `train_loss`: Training data loss
- `train_quality`: Training R² (regression) or accuracy (classification)
- `val_loss`: Validation data loss
- `val_quality`: Validation R² or accuracy
- `reg_loss`: Regularized loss (loss + jitter × ||weights||²)
- `jitter`: Current jitter value
- `weight_norm_sq`: Squared Frobenius norm of weights

**Per Epoch (GTN):**
- `train_loss`: Training loss
- `val_loss`: Validation loss
- `val_quality`: Validation R² or accuracy

**Summary (Final):**
- `seed`: Random seed
- `train_loss`, `train_quality`: Final training metrics
- `val_loss`, `val_quality`: Best validation metrics
- `test_loss`, `test_quality`: Test set metrics
- `best_epoch`: Epoch with best validation
- `success`: True if training succeeded

---

## Complete Parameter Reference

### Model Architecture Parameters

All three models (MPO2, LMPO2, MMPO2) share these base parameters:

| Parameter | Type | Range | Default | Description |
|-----------|------|-------|---------|-------------|
| `model` | string | MPO2/LMPO2/MMPO2 | MPO2 | Model type |
| `L` | int | 2-10 | 3 | Number of tensor sites |
| `bond_dim` | int | 2-20 | 6 | Bond dimension between tensors |
| `phys_dim` | int | auto | from data | Physical (input) dimension |
| `output_dim` | int | auto | from data | Output dimension |
| `output_site` | int | 0 to L-1 | 1 | Which site carries output |
| `init_strength` | float | 0.0001-1.0 | 0.001 | Initialization scale |
| `use_tn_normalization` | bool | true/false | true | Frobenius normalization |

**LMPO2-specific:**
| Parameter | Type | Range | Default | Description |
|-----------|------|-------|---------|-------------|
| `rank` | int | 2-20 | 5 | Reduced dimension |
| `reduction_factor` | float | 0.1-0.9 | 0.5 | Alternative to rank |

**MMPO2-specific:**
| Parameter | Type | Range | Default | Description |
|-----------|------|-------|---------|-------------|
| `rank` | int | 2-20 | 5 | Mask complexity |

### NTN Training Parameters

| Parameter | Type | Range | Default | Description |
|-----------|------|-------|---------|-------------|
| `n_epochs` | int | 5-100 | 10-50 | Number of training epochs |
| `batch_size` | int | 10-500 | 100 | Batch size |
| `jitter_start` | float | 1e-6 to 10.0 | 0.001 (reg), 5.0 (class) | Initial ridge penalty |
| `jitter_decay` | float | 0.5-0.99 | 0.95 | Jitter decay per epoch |
| `jitter_min` | float | 1e-8 to 1e-3 | 0.001 | Minimum jitter |
| `adaptive_jitter` | bool | true/false | true | Auto-adjust jitter |
| `patience` | int/null | 3-20/null | 5 | Early stopping patience |
| `min_delta` | float | 0.0001-0.01 | 0.001 | Min improvement for early stop |
| `train_selection` | bool | true/false | false | Use train quality fallback |

**Jitter schedule:**
```
jitter[epoch] = max(jitter_start × jitter_decay^epoch, jitter_min)
```

### GTN Training Parameters

| Parameter | Type | Range | Default | Description |
|-----------|------|-------|---------|-------------|
| `n_epochs` | int | 30-500 | 100 | Number of training epochs |
| `batch_size` | int | 16-256 | 32 | Mini-batch size |
| `lr` | float | 1e-5 to 0.1 | 0.001 | Learning rate |
| `weight_decay` | float | 0.0-1.0 | 0.01 | L2 regularization |
| `optimizer` | string | adam/adamw/sgd | adam | Optimizer type |
| `loss_fn` | string | mse/mae/huber/cross_entropy | auto | Loss function |

**Optimizer characteristics:**
- **adam**: Adaptive learning rates, good default
- **adamw**: Adam with decoupled weight decay, better for generalization
- **sgd**: Simple gradient descent with momentum=0.9

**Loss functions:**
- **mse**: Mean Squared Error (L2 loss)
- **mae**: Mean Absolute Error (L1 loss)
- **huber**: Huber loss (combines L1 and L2)
- **cross_entropy**: Cross-entropy (classification)

### Dataset Configuration

| Parameter | Type | Options | Description |
|-----------|------|---------|-------------|
| `dataset` | string | See section above | Dataset name |
| `task` | string | regression/classification | Task type (auto-inferred) |

### Experiment Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `experiment_name` | string | dataset name | Experiment identifier |
| `seeds` | list[int] | [0,1,2,3,4] | Random seeds for reproducibility |
| `verbose` | bool | false | Print training progress |

### Tracking Configuration

| Parameter | Type | Options | Description |
|-----------|------|---------|-------------|
| `backend` | string | file/aim/both/none | Tracking backend |
| `tracker_dir` | string | experiment_logs | Directory for file logs |
| `aim_repo` | string/null | aim://host:port | AIM server URL |

### Output Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `results_dir` | string | results/{experiment_name} | Output directory |
| `save_models` | bool | false | Save model weights (not implemented) |
| `save_individual_runs` | bool | true | Save per-run JSON files |

---

## Best Practices & Tuning

### Hyperparameter Tuning Strategy

**Phase 1: Baseline (5 minutes)**

Start with a single configuration to verify everything works:

```json
{
  "parameter_grid": {
    "model": ["MPO2"],
    "bond_dim": [6]
  },
  "fixed_params": {
    "L": 3,
    "seeds": [0]
  }
}
```

**Phase 2: Coarse Search (30 minutes - 1 hour)**

Test broad ranges to find promising regions:

```json
{
  "parameter_grid": {
    "bond_dim": [4, 6, 8, 10],
    "jitter_start": [0.001, 0.01, 0.1, 1.0, 10.0]
  },
  "fixed_params": {
    "seeds": [0, 1, 2]
  }
}
```

**Phase 3: Fine-tune (2-4 hours)**

Zoom in on best regions with more seeds:

```json
{
  "parameter_grid": {
    "bond_dim": [6, 7, 8],
    "jitter_start": [0.5, 1.0, 2.0, 5.0, 10.0],
    "L": [2, 3, 4]
  },
  "fixed_params": {
    "seeds": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
  }
}
```

### Seeds and Reproducibility

**Always use multiple seeds:**
```json
{
  "fixed_params": {
    "seeds": [0, 1, 2, 3, 4]
  }
}
```

**Report statistics:**
- Mean ± standard deviation across seeds
- Minimum and maximum values
- Confidence intervals if needed

**Example reporting:**
```
MPO2 (bond_dim=8, jitter=5.0):
  Test Accuracy: 87.2 ± 2.3%
  Range: [83.5%, 91.3%]
  n=5 seeds
```
---

## API Reference

### Command-Line Scripts

**NTN Individual Training:**
```bash
python experiments/train_mpo2.py --help
python experiments/train_lmpo2.py --help
python experiments/train_mmpo2.py --help
```

**GTN Individual Training:**
```bash
python experiments/gtn/train_mpo2_gtn.py --help
```

**Grid Search:**
```bash
python experiments/run_grid_search.py --help       # NTN
python experiments/run_grid_search_gtn.py --help   # GTN
```

### Python API

**Dataset Loading:**
```python
from experiments.dataset_loader import load_dataset

data, info = load_dataset('iris')

X_train, y_train = data['X_train'], data['y_train']
X_val, y_val = data['X_val'], data['y_val']
X_test, y_test = data['X_test'], data['y_test']

print(f"Dataset: {info['name']}")
print(f"Task: {info['task']}")
print(f"Samples: train={info['n_train']}, val={info['n_val']}, test={info['n_test']}")
```

**Model Creation:**
```python
from model.MPO2_models import MPO2, LMPO2, MMPO2

model = MPO2(L=3, bond_dim=8, phys_dim=4, output_dim=3)

model_lmpo = LMPO2(L=3, bond_dim=8, phys_dim=100, output_dim=10, rank=5)

model_mmpo = MMPO2(L=4, bond_dim=6, phys_dim=20, output_dim=5, rank=5)
```

**NTN Training:**
```python
from model.NTN import NTN
from model.losses import MSELoss, CrossEntropyLoss
from model.utils import create_inputs

loss_fn = CrossEntropyLoss()

loader_train = create_inputs(
    X=X_train, y=y_train,
    input_labels=model.input_labels,
    output_labels=model.output_dims,
    batch_size=100,
    append_bias=False
)

loader_val = create_inputs(
    X=X_val, y=y_val,
    input_labels=model.input_labels,
    output_labels=model.output_dims,
    batch_size=100,
    append_bias=False
)

ntn = NTN(
    tn=model.tn,
    output_dims=model.output_dims,
    input_dims=model.input_dims,
    loss=loss_fn,
    data_stream=loader_train
)

jitter_schedule = [max(5.0 * (0.95 ** epoch), 0.001) for epoch in range(20)]

scores_train, scores_val = ntn.fit(
    val_data=loader_val,
    n_epochs=20,
    jitter_schedule=jitter_schedule,
    adaptive_jitter=True,
    patience=5,
    verbose=True
)

print(f"Final validation accuracy: {scores_val['acc']:.4f}")
```

**GTN Training:**
```python
from model.GTN import GTN
import torch
import torch.nn as nn
import torch.optim as optim
import quimb.tensor as qt

class MPO2GTN(GTN):
    def construct_nodes(self, x):
        return [qt.Tensor(x, inds=["s", label], tags=f"Input_{label}") 
                for label in self.input_dims]

gtn = MPO2GTN(
    tn=model.tn,
    output_dims=["s", "out"],
    input_dims=model.input_dims
)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(gtn.parameters(), lr=0.001, weight_decay=0.01)

train_loader = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(X_train, y_train),
    batch_size=32,
    shuffle=True
)

val_loader = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(X_val, y_val),
    batch_size=32,
    shuffle=False
)

for epoch in range(100):
    gtn.train()
    train_loss = 0.0
    
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        output = gtn(X_batch)
        loss = criterion(output, y_batch)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    
    gtn.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            output = gtn(X_batch)
            loss = criterion(output, y_batch)
            val_loss += loss.item()
            
            pred = output.argmax(dim=1)
            target = y_batch.argmax(dim=1)
            correct += (pred == target).sum().item()
            total += y_batch.size(0)
    
    val_acc = correct / total
    print(f"Epoch {epoch+1}: Val Loss={val_loss/len(val_loader):.4f}, Val Acc={val_acc:.4f}")
```

**Configuration Parser:**
```python
from experiments.config_parser import load_config, create_experiment_plan

config = load_config('experiments/configs/iris_minimal_test.json')
experiments, metadata = create_experiment_plan(config)

print(f"Total experiments: {metadata['total_experiments']}")
print(f"Grid combinations: {metadata['grid_size']}")
print(f"Seeds per combination: {metadata['n_seeds']}")

for exp in experiments[:5]:
    print(f"Run ID: {exp['run_id']}")
    print(f"  Model: {exp['params']['model']}")
    print(f"  Bond dim: {exp['params']['bond_dim']}")
    print(f"  Seed: {exp['seed']}")
```

**Tracking:**
```python
from experiments.trackers import create_tracker

tracker = create_tracker(
    experiment_name="iris",
    config={"model": "MPO2", "bond_dim": 8, "seed": 0},
    backend="file",
    output_dir="experiment_logs",
    repo=None
)

tracker.log_hparams({"model": "MPO2", "bond_dim": 8})

for epoch in range(10):
    metrics = {
        "train_loss": 0.5 - epoch * 0.03,
        "val_loss": 0.6 - epoch * 0.02,
        "val_quality": 0.7 + epoch * 0.02
    }
    tracker.log_metrics(metrics, step=epoch)

tracker.log_summary({"test_quality": 0.87, "success": True})
tracker.close()
```
