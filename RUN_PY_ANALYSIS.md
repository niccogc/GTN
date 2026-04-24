# Comprehensive Analysis of run.py

## Overview
`run.py` is the unified experiment runner for tensor network models using Hydra configuration. It orchestrates:
1. **Model instantiation** (NTN or GTN variants)
2. **Dataset loading** from UCI ML Repository
3. **Training/testing** via Newton-based (NTN) or gradient-based (GTN) optimization
4. **Result tracking** and experiment management

---

## 1. HOW MODELS ARE INSTANTIATED AND CONFIGURED

### 1.1 Configuration System (Hydra)

**Entry Point:** `@hydra.main(version_base=None, config_path="conf", config_name="config")`

**Config Hierarchy:**
```
conf/
├── config.yaml              # Main config with defaults
├── model/                   # Model configs
│   ├── _base.yaml          # Base model settings
│   ├── mpo2.yaml           # MPO2 model
│   ├── lmpo2.yaml          # LMPO2 with reduction
│   ├── mmpo2.yaml          # MMPO2
│   ├── cpda.yaml           # CPDA
│   ├── mpo2_typei.yaml     # TypeI variants
│   ├── tnml_p.yaml         # TNML polynomial
│   └── tnml_f.yaml         # TNML Fourier
├── trainer/                # Training configs
│   ├── ntn.yaml            # Newton-based training
│   └── gtn.yaml            # Gradient-based training
└── dataset/                # Dataset configs
    ├── _base.yaml
    ├── iris.yaml           # Small classification
    ├── abalone.yaml        # Large regression
    └── [20 other datasets]
```

**Default Configuration (config.yaml):**
```yaml
defaults:
  - model: mpo2           # Default model
  - dataset: iris         # Default dataset (small, classification)
  - trainer: ntn          # Default trainer (Newton-based)

seed: 42
skip_completed: true      # Skip if results.json exists
update_tracking: false    # Set true on main machine only
```

### 1.2 Model Instantiation Flow

**Step 1: Load Configuration**
```python
@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    # cfg contains merged config from all YAML files
    # Example: cfg.model.name = "MPO2", cfg.model.bond_dim = 6, etc.
```

**Step 2: Create Model Instance**
```python
# Line 670 in run.py
model = create_model(cfg, input_dim, output_dim, raw_feature_count=raw_feature_count)
```

**Step 3: Model Creation Function (Lines 217-224)**
```python
def create_model(cfg: DictConfig, input_dim: int, output_dim: int, raw_feature_count: int = None):
    """Create model instance from config."""
    model_cls = NTN_MODELS.get(cfg.model.name)  # Look up model class
    if model_cls is None:
        raise ValueError(f"Unknown model: {cfg.model.name}")
    
    params = build_model_params(cfg, input_dim, output_dim, for_gtn=False, raw_feature_count=raw_feature_count)
    return model_cls(**params)
```

### 1.3 Model Registry (Lines 126-144)

**NTN Models (Newton-based):**
```python
NTN_MODELS = {
    "MPO2": MPO2,              # Simple MPS
    "LMPO2": LMPO2,            # Linear MPO2 (with dimensionality reduction)
    "MMPO2": MMPO2,            # Modified MPO2
    "CPDA": CPDA,              # CP Decomposition
    "MPO2TypeI": MPO2TypeI,    # Variable-site ensemble
    "LMPO2TypeI": LMPO2TypeI,
    "MMPO2TypeI": MMPO2TypeI,
    "CPDATypeI": CPDATypeI,
    "TNML_P": TNML_P,          # Tensor Network ML with polynomial encoding
    "TNML_F": TNML_F,          # Tensor Network ML with Fourier encoding
}

GTN_TYPEI_MODELS = {
    "MPO2TypeI": MPO2TypeI_GTN,    # GTN variants for TypeI models
    "LMPO2TypeI": LMPO2TypeI_GTN,
    "MMPO2TypeI": MMPO2TypeI_GTN,
    "CPDATypeI": CPDATypeI_GTN,
}
```

### 1.4 Parameter Building (Lines 162-186)

```python
def build_model_params(cfg, input_dim, output_dim, for_gtn=False, raw_feature_count=None):
    """Build model parameters from config."""
    is_typei = cfg.model.name.endswith("TypeI")
    is_tnml = cfg.model.name.startswith("TNML")
    
    params = {
        "phys_dim": raw_feature_count if is_tnml else input_dim,
        "output_dim": output_dim,
        "output_site": cfg.model.get("output_site"),
        "init_strength": cfg.model.get("init_strength", 0.001 if for_gtn else 0.1),
        "bond_dim": cfg.model.bond_dim,
        "L": cfg.model.L,
    }
    
    # TypeI models use max_sites instead of L
    if is_typei:
        params["max_sites"] = params.pop("L")
    
    # LMPO2 needs reduction dimension
    if "LMPO2" in cfg.model.name:
        params["reduced_dim"] = get_reduced_dim(cfg, input_dim)
        if not is_typei and not for_gtn:
            params["bond_dim_mpo"] = cfg.model.get("bond_dim_mpo", 2)
    
    return params
```

---

## 2. HOW INPUTS ARE PREPARED FOR MODELS

### 2.1 Dataset Loading (Lines 653-666)

```python
# Load dataset
log.info(f"Loading dataset: {cfg.dataset.name}")
data, dataset_info = load_dataset(cfg.dataset.name)
data = move_data_to_device(data)

raw_feature_count = data["X_train"].shape[1]
input_dim = raw_feature_count + 1  # +1 for bias term
output_dim = data["y_train"].shape[1] if data["y_train"].ndim > 1 else 1
```

**Returned Data Structure:**
```python
data = {
    "X_train": torch.Tensor,  # (n_train, n_features)
    "y_train": torch.Tensor,  # (n_train,) or (n_train, n_classes)
    "X_val": torch.Tensor,
    "y_val": torch.Tensor,
    "X_test": torch.Tensor,
    "y_test": torch.Tensor,
}

dataset_info = {
    "name": "iris",
    "task": "classification",  # or "regression"
    "n_train": 105,
    "n_val": 22,
    "n_test": 23,
    "n_features": 4,
    "n_classes": 3,  # if classification
}
```

### 2.2 Input Preparation for NTN (Lines 274-295)

**For Standard Models (MPO2, LMPO2, etc.):**
```python
encoding = getattr(model, "encoding", None)
poly_degree = getattr(model, "poly_degree", None) if encoding == "polynomial" else None

loader_train = create_inputs(
    X=data["X_train"],
    y=data["y_train"],
    input_labels=model.input_labels,      # e.g., ["x0", "x1", "x2"]
    output_labels=model.output_dims,      # e.g., ["out"]
    batch_size=cfg.dataset.batch_size,
    append_bias=(encoding is None),       # Add bias term if no encoding
    encoding=encoding,                    # None, "polynomial", or "fourier"
    poly_degree=poly_degree,
)

loader_val = create_inputs(...)  # Same for validation
```

---

## 3. HOW TRAINING/TESTING IS DONE

### 3.1 Trainer Selection (Lines 673-678)

```python
if cfg.trainer.type == "ntn":
    result = run_ntn(cfg, model, data, output_dir)
elif cfg.trainer.type == "gtn":
    result = run_gtn(cfg, model, data, output_dir)
else:
    raise ValueError(f"Unknown trainer type: {cfg.trainer.type}")
```

### 3.2 NTN Training (Newton-based, Lines 227-400)

**Configuration (conf/trainer/ntn.yaml):**
```yaml
trainer:
  type: ntn
  n_epochs: 20
  ridge: 5                    # Regularization strength
  ridge_decay: 0.25           # Multiplicative decay per epoch
  ridge_min: 0.0001
  adaptive_ridge: false
  patience: 10                # Early stopping
  min_delta: 0.001
  train_selection: true       # Use training quality as tiebreaker
```

### 3.3 GTN Training (Gradient-based, Lines 403-593)

**Configuration (conf/trainer/gtn.yaml):**
```yaml
trainer:
  type: gtn
  n_epochs: 1000
  lr: 0.005                   # Learning rate
  optimizer: adamw            # adam, adamw, sgd
  ridge: 0.005                # weight_decay = 2 * ridge
  loss_fn: null               # null=auto, mse, mae, huber, cross_entropy
  patience: 100
  min_delta: 0.001
  train_selection: true
```

---

## 4. AVAILABLE DATASETS (ESPECIALLY SMALL REGRESSION)

### 4.1 Dataset Registry (model/load_ucirepo.py, Lines 11-33)

**All Available Datasets:**
```python
datasets = [
    # REGRESSION DATASETS
    ("student_perf", 320, "regression"),      # Large
    ("abalone", 1, "regression"),             # Large
    ("obesity", 544, "regression"),           # Medium
    ("bike", 275, "regression"),              # Medium
    ("realstate", 477, "regression"),         # Medium
    ("energy_efficiency", 242, "regression"), # Medium
    ("concrete", 165, "regression"),          # Medium
    ("ai4i", 601, "regression"),              # Large
    ("appliances", 374, "regression"),        # Large
    ("popularity", 332, "regression"),        # Large
    ("seoulBike", 560, "regression"),         # Medium
    
    # CLASSIFICATION DATASETS
    ("iris", 53, "classification"),           # Small
    ("hearth", 45, "classification"),         # Small
    ("winequalityc", 186, "classification"),  # Medium
    ("breast", 17, "classification"),         # Small
    ("adult", 2, "classification"),           # Large
    ("bank", 222, "classification"),          # Medium
    ("wine", 109, "classification"),          # Small
    ("car_evaluation", 19, "classification"), # Small
    ("student_dropout", 697, "classification"),  # Large
    ("mushrooms", 73, "classification"),      # Large
]
```

### 4.2 Small Regression Datasets

**NONE EXIST!** All regression datasets are medium or large.

**Smallest Regression Datasets:**
1. **concrete** (165 samples) - Medium
2. **energy_efficiency** (242 samples) - Medium
3. **bike** (275 samples) - Medium

### 4.3 Small Classification Datasets (for reference)

```
iris (150 samples)
hearth (303 samples)
breast (569 samples)
wine (178 samples)
car_evaluation (1728 samples)
```

---

## 5. HOW TO SPECIFY MODEL TYPE IN CONFIG OR COMMAND LINE

### 5.1 Command Line Overrides

**Basic Usage:**
```bash
# Use defaults (MPO2, iris, NTN)
python run.py

# Override model
python run.py model=lmpo2
python run.py model=mmpo2
python run.py model=cpda
python run.py model=mpo2_typei
python run.py model=tnml_p
python run.py model=tnml_f

# Override dataset
python run.py dataset=abalone
python run.py dataset=concrete
python run.py dataset=wine

# Override trainer
python run.py trainer=gtn
python run.py trainer=ntn

# Combine overrides
python run.py model=lmpo2 dataset=abalone trainer=gtn

# Override specific parameters
python run.py model.bond_dim=8
python run.py model.L=5
python run.py trainer.lr=0.01
python run.py trainer.ridge=10
python run.py seed=123

# Grid search (multirun)
python run.py --multirun model.bond_dim=4,6,8
python run.py --multirun seed=0,1,2,3,4
python run.py --multirun model=mpo2,lmpo2,mmpo2 dataset=iris,wine
```

### 5.2 Configuration File Approach

**Create custom config file (e.g., conf/experiment/my_experiment.yaml):**
```yaml
# @package _global_
defaults:
  - model: lmpo2
  - dataset: concrete
  - trainer: gtn

model:
  bond_dim: 8
  L: 4
  reduction_factor: 0.5

trainer:
  lr: 0.01
  n_epochs: 500
  patience: 50

seed: 42
```

**Run with custom config:**
```bash
python run.py --config-name=my_experiment
```

### 5.3 Model-Specific Parameters

**MPO2 (Simple MPS):**
```bash
python run.py model=mpo2 model.bond_dim=6 model.L=3 model.init_strength=0.1
```

**LMPO2 (With Reduction):**
```bash
python run.py model=lmpo2 model.reduction_factor=0.3 model.bond_dim_mpo=1
# OR specify reduced_dim directly
python run.py model=lmpo2 model.reduced_dim=5
```

**TNML_P (Polynomial Encoding):**
```bash
python run.py model=tnml_p model.poly_degree=3
```

**TNML_F (Fourier Encoding):**
```bash
python run.py model=tnml_f
```

**TypeI Models (Variable Sites):**
```bash
python run.py model=mpo2_typei model.max_sites=5
```

### 5.4 Trainer-Specific Parameters

**NTN (Newton-based):**
```bash
python run.py trainer=ntn trainer.ridge=5 trainer.ridge_decay=0.25 trainer.patience=10
```

**GTN (Gradient-based):**
```bash
python run.py trainer=gtn trainer.lr=0.005 trainer.optimizer=adamw trainer.patience=100
```

### 5.5 Dataset-Specific Parameters

```bash
python run.py dataset=iris dataset.batch_size=32
python run.py dataset=abalone dataset.batch_size=128
```

---

## 6. FULL FLOW FROM run.py TO MODEL FORWARD PASS

### 6.1 Complete Execution Flow

```
1. INITIALIZATION
   ├─ @hydra.main() loads config from conf/
   ├─ Merges defaults: model=mpo2, dataset=iris, trainer=ntn
   └─ Sets seed: torch.manual_seed(42), np.random.seed(42)

2. DATASET LOADING
   ├─ load_dataset("iris")
   ├─ Fetch from UCI ML Repository
   ├─ One-hot encode categoricals
   ├─ Split: 70% train, 15% val, 15% test
   ├─ Standardize numeric features
   └─ Return: data = {X_train, y_train, X_val, y_val, X_test, y_test}

3. MODEL INSTANTIATION
   ├─ create_model(cfg, input_dim=5, output_dim=3)
   ├─ Look up: NTN_MODELS["MPO2"] = MPO2 class
   ├─ Build params: {phys_dim=5, output_dim=3, bond_dim=6, L=3, ...}
   ├─ Instantiate: model = MPO2(**params)
   │  └─ Creates tensor network with quimb
   │     ├─ L=3 sites with bond_dim=6
   │     ├─ Physical dimension = 5 (4 features + 1 bias)
   │     ├─ Output dimension = 3 (3 classes)
   │     └─ Tags: Node0, Node1, Node2
   └─ Extract: model.tn, model.input_labels, model.output_dims

4. INPUT PREPARATION
   ├─ For NTN:
   │  ├─ create_inputs(X_train, y_train, input_labels=["x0", "x1", "x2"], ...)
   │  ├─ Append bias: X_train (105, 4) -> (105, 5)
   │  ├─ Create Inputs object with batches
   │  └─ Each batch: (mu=[qt.Tensor, ...], y=qt.Tensor)
   │
   └─ For GTN:
      ├─ Append bias: X_train (105, 4) -> (105, 5)
      ├─ Create PyTorch DataLoader
      └─ Each batch: (X_batch, y_batch) as torch.Tensor

5. TRAINER SELECTION
   ├─ if cfg.trainer.type == "ntn":
   │  └─ run_ntn(cfg, model, data, output_dir)
   │
   └─ elif cfg.trainer.type == "gtn":
      └─ run_gtn(cfg, model, data, output_dir)

6. NTN TRAINING (if selected)
   ├─ Create NTN optimizer:
   │  └─ ntn = NTN(tn=model.tn, output_dims=["out"], input_dims=["x0", "x1", "x2"], ...)
   │
   ├─ Create ridge schedule: [5.0, 1.25, 0.3125, ...]
   │
   ├─ For each epoch:
   │  ├─ For each batch (mu, y):
   │  │  ├─ Forward pass: ntn.forward(mu)
   │  │  │  ├─ Contract tensor network with inputs
   │  │  │  └─ Return predictions
   │  │  ├─ Compute loss: loss_fn(pred, y)
   │  │  ├─ Compute gradient: grad = ∇loss
   │  │  ├─ Compute Hessian: hess = ∇²loss
   │  │  └─ Update: θ = θ - (hess + ridge*I)^{-1} @ grad
   │  │
   │  ├─ Evaluate on validation set
   │  └─ Log metrics
   │
   └─ Return: best_train_loss, best_val_quality, metrics_log

7. GTN TRAINING (if selected)
   ├─ Create GTN model:
   │  └─ gtn_model = GTN(tn=model.tn, output_dims=["out"], input_dims=["x0", "x1", "x2"])
   │     ├─ Pack tensor network into PyTorch parameters
   │     └─ Move to GPU if available
   │
   ├─ Create optimizer: optim.AdamW(gtn_model.parameters(), lr=0.005, weight_decay=0.01)
   │
   ├─ For each epoch:
   │  ├─ For each batch (X_batch, y_batch):
   │  │  ├─ Forward pass: output = gtn_model(X_batch)
   │  │  │  ├─ Unpack parameters from PyTorch
   │  │  │  ├─ Construct input tensor nodes
   │  │  │  ├─ Contract: tn & input_tn
   │  │  │  └─ Return output
   │  │  ├─ Compute loss: loss = criterion(output, y_batch)
   │  │  ├─ Backward: loss.backward()
   │  │  └─ Optimizer step: optimizer.step()
   │  │
   │  ├─ Evaluate on validation set
   │  └─ Log metrics
   │
   └─ Return: best_train_loss, best_val_quality, metrics_log

8. RESULT SAVING
   ├─ Save results.json with:
   │  ├─ success, singular, train_loss, val_quality, best_epoch
   │  ├─ metrics_log (per-epoch metrics)
   │  ├─ config (full configuration)
   │  └─ dataset_info
   │
   └─ Update tracking CSV (if enabled)
```

---

## 7. QUICK REFERENCE: COMMON COMMANDS

### Run with Different Models
```bash
python run.py model=mpo2 dataset=iris trainer=ntn
python run.py model=lmpo2 dataset=concrete trainer=gtn
python run.py model=cpda dataset=wine trainer=ntn
python run.py model=tnml_p dataset=iris trainer=gtn
python run.py model=mpo2_typei dataset=abalone trainer=ntn
```

### Grid Search
```bash
python run.py --multirun model.bond_dim=4,6,8,10
python run.py --multirun seed=0,1,2,3,4
python run.py --multirun model=mpo2,lmpo2,mmpo2 trainer=ntn,gtn
```

### Custom Hyperparameters
```bash
python run.py model=lmpo2 model.reduction_factor=0.5 trainer=gtn trainer.lr=0.01
python run.py model=mpo2 model.L=5 trainer=ntn trainer.ridge=10
```

### Debug Mode
```bash
python run.py model=mpo2 dataset=iris trainer=ntn seed=42 skip_completed=false
```

---

## 8. KEY INSIGHTS

1. **Model Types:**
   - **Standard Models** (MPO2, LMPO2, MMPO2, CPDA): Fixed number of sites
   - **TypeI Models** (MPO2TypeI, etc.): Variable number of sites (ensemble)
   - **TNML Models** (TNML_P, TNML_F): Feature encoding (polynomial/Fourier)

2. **Training Methods:**
   - **NTN**: Second-order optimization, fewer epochs (20), ridge regularization
   - **GTN**: First-order gradient descent, more epochs (1000), standard optimizers

3. **Input Handling:**
   - **Standard**: Append bias term to features
   - **TNML**: Encode features (polynomial or Fourier) into higher-dimensional space

4. **Dataset Splits:**
   - Fixed 70/15/15 split (seed=42) for reproducibility
   - Experiment seed (default 42) controls model initialization, not data splits

5. **No Small Regression Datasets:**
   - Smallest regression dataset is "concrete" with 165 samples
   - For small datasets, use classification datasets (iris, wine, breast, etc.)

