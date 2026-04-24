# run.py Code Examples and Deep Dives

## 1. DETAILED MODEL INSTANTIATION EXAMPLES

### Example 1: MPO2 (Simple MPS)

**Configuration:**
```yaml
# conf/model/mpo2.yaml
model:
  name: MPO2
  L: 3
  bond_dim: 6
  init_strength: 0.1
```

**Instantiation Code:**
```python
# In run.py, line 670
model = create_model(cfg, input_dim=5, output_dim=3)

# This calls:
def create_model(cfg, input_dim=5, output_dim=3, raw_feature_count=4):
    model_cls = NTN_MODELS.get("MPO2")  # Gets MPO2 class
    
    params = build_model_params(cfg, input_dim=5, output_dim=3, for_gtn=False, raw_feature_count=4)
    # params = {
    #     "phys_dim": 5,           # 4 features + 1 bias
    #     "output_dim": 3,         # 3 classes
    #     "output_site": None,     # Will default to L-1 = 2
    #     "init_strength": 0.1,
    #     "bond_dim": 6,
    #     "L": 3,
    # }
    
    return MPO2(**params)

# MPO2.__init__ creates:
# - Node0: (phys_dim=5, bond_dim=6) with indices (x0, b0)
# - Node1: (bond_dim=6, phys_dim=5, bond_dim=6) with indices (b0, x1, b1)
# - Node2: (bond_dim=6, phys_dim=5, output_dim=3) with indices (b1, x2, out)
# - Tags: {Node0}, {Node1}, {Node2}
```

**Resulting Tensor Network:**
```
Input features: [f0, f1, f2, f3, bias]
                  ↓   ↓   ↓   ↓   ↓
                [x0] [x1] [x2]
                  |    |    |
    ┌─────────────┴────┴────┴──────────────┐
    │                                       │
  Node0 ─── b0 ─── Node1 ─── b1 ─── Node2 ─── out
    │                │                │
   x0               x1               x2
                                      │
                                   [out]
                                      ↓
                                  [3 classes]
```

### Example 2: LMPO2 (With Dimensionality Reduction)

**Configuration:**
```yaml
# conf/model/lmpo2.yaml
model:
  name: LMPO2
  L: 3
  bond_dim: 6
  reduction_factor: 0.3      # Reduce to 30% of input dimension
  bond_dim_mpo: 1
```

**Instantiation Code:**
```python
model = create_model(cfg, input_dim=5, output_dim=3, raw_feature_count=4)

# build_model_params detects LMPO2:
params = {
    "phys_dim": 5,
    "output_dim": 3,
    "output_site": None,
    "init_strength": 0.1,
    "bond_dim": 6,
    "L": 3,
    "reduced_dim": get_reduced_dim(cfg, input_dim=5),  # max(2, int(5 * 0.3)) = 2
    "bond_dim_mpo": 1,
}

# LMPO2.__init__ creates two layers:
# MPO layer (dimensionality reduction):
#   - MPO0: (phys_dim=5, reduced_dim=2, bond_dim_mpo=1) with indices (0_in, 0_reduced, b_mpo_0)
#   - MPO1: (bond_dim_mpo=1, phys_dim=5, reduced_dim=2, bond_dim_mpo=1) with indices (b_mpo_0, 1_in, 1_reduced, b_mpo_1)
#   - MPO2: (bond_dim_mpo=1, phys_dim=5, reduced_dim=2) with indices (b_mpo_1, 2_in, 2_reduced)
#
# MPS layer (output):
#   - MPS0: (reduced_dim=2, bond_dim=6) with indices (0_reduced, b0)
#   - MPS1: (bond_dim=6, reduced_dim=2, bond_dim=6) with indices (b0, 1_reduced, b1)
#   - MPS2: (bond_dim=6, reduced_dim=2, output_dim=3) with indices (b1, 2_reduced, out)
```

**Data Flow:**
```
Input: [f0, f1, f2, f3, bias] (5 features)
         ↓   ↓   ↓   ↓   ↓
       MPO layer (reduction)
         ↓   ↓   ↓   ↓   ↓
       [r0, r1, r2] (3 reduced features, each dim=2)
         ↓   ↓   ↓
       MPS layer (output)
         ↓   ↓   ↓
       [3 classes]
```

### Example 3: TNML_P (Polynomial Encoding)

**Configuration:**
```yaml
# conf/model/tnml_p.yaml
model:
  name: TNML_P
  L: 3
  bond_dim: 6
  poly_degree: 3  # [1, x, x^2, x^3]
```

**Instantiation Code:**
```python
model = create_model(cfg, input_dim=5, output_dim=3, raw_feature_count=4)

# build_model_params detects TNML:
params = {
    "phys_dim": 4,  # raw_feature_count (NOT input_dim with bias!)
    "output_dim": 3,
    "output_site": None,
    "init_strength": 0.1,
    "bond_dim": 6,
    "L": 3,
    "poly_degree": 3,
}

# TNML_P.__init__ creates:
# - Node0: (phys_dim=4, bond_dim=6) with indices (x0, b0)
# - Node1: (bond_dim=6, phys_dim=4, bond_dim=6) with indices (b0, x1, b1)
# - Node2: (bond_dim=6, phys_dim=4, output_dim=3) with indices (b1, x2, out)
```

**Input Encoding:**
```
Raw input: [f0, f1, f2, f3] (4 features)
           ↓   ↓   ↓   ↓
Polynomial encoding (degree=3):
           ↓   ↓   ↓   ↓
[1, f0, f0², f0³] [1, f1, f1², f1³] [1, f2, f2², f2³] [1, f3, f3², f3³]
(4 features × 4 polynomial terms = 4 input nodes, each with phys_dim=4)
           ↓   ↓   ↓   ↓
       MPS layer
           ↓   ↓   ↓   ↓
       [3 classes]
```

### Example 4: MPO2TypeI (Variable Sites Ensemble)

**Configuration:**
```yaml
# conf/model/mpo2_typei.yaml
model:
  name: MPO2TypeI
  L: 3              # This becomes max_sites for TypeI
  bond_dim: 6
```

**Instantiation Code:**
```python
model = create_model(cfg, input_dim=5, output_dim=3, raw_feature_count=4)

# build_model_params detects TypeI:
params = {
    "phys_dim": 5,
    "output_dim": 3,
    "output_site": None,
    "init_strength": 0.1,
    "bond_dim": 6,
    "max_sites": 3,  # Changed from L to max_sites
}

# MPO2TypeI.__init__ creates an ensemble:
# - model.tns = [tn_1site, tn_2site, tn_3site]
# - Each tn_i has i sites with bond_dim=6
# - During training, all are trained simultaneously
```

---

## 2. DETAILED INPUT PREPARATION EXAMPLES

### Example 1: Standard Model Input Preparation (NTN)

**Raw Data:**
```python
X_train = torch.tensor([
    [0.5, -0.2, 1.3, 0.8],  # Sample 1
    [0.1,  0.3, 0.9, -0.5], # Sample 2
    ...
], dtype=torch.float64)  # Shape: (105, 4)

y_train = torch.tensor([
    [1, 0, 0],  # Sample 1: class 0
    [0, 1, 0],  # Sample 2: class 1
    ...
], dtype=torch.float64)  # Shape: (105, 3) - one-hot encoded
```

**Input Preparation:**
```python
loader_train = create_inputs(
    X=X_train,
    y=y_train,
    input_labels=["x0", "x1", "x2"],  # From model.input_labels
    output_labels=["out"],             # From model.output_dims
    batch_size=32,
    append_bias=True,                  # Add bias term
    encoding=None,
)

# Inside create_inputs():
# 1. Append bias:
X_with_bias = torch.cat([X_train, torch.ones(105, 1)], dim=1)
# Shape: (105, 5)

# 2. Create Inputs object:
# inputs_list = [X_with_bias]  # Single input tensor
# outputs_list = [y_train]

# 3. Create batches:
# For batch 0 (samples 0-31):
#   input_dict = {0: X_with_bias[0:32]}  # (32, 5)
#   y_tensor = y_train[0:32]              # (32, 3)
#
#   _prepare_batch(input_dict) creates:
#   mu = [
#       qt.Tensor(X_with_bias[0:32], inds=["s", "x0"], tags={"input_x0", "I0"}),
#       qt.Tensor(X_with_bias[0:32], inds=["s", "x1"], tags={"input_x1", "I1"}),
#       qt.Tensor(X_with_bias[0:32], inds=["s", "x2"], tags={"input_x2", "I2"}),
#   ]
#
#   batch = (mu, y_tensor)
```

**Batch Structure:**
```python
# loader_train[0] returns:
mu, y = loader_train[0]

# mu is a list of 3 quimb tensors:
mu[0].data.shape  # (32, 5) - batch of 32 samples, 5 features
mu[0].inds        # ("s", "x0")
mu[0].tags        # {"input_x0", "I0"}

# y is a quimb tensor:
y.data.shape      # (32, 3) - batch of 32 samples, 3 classes
y.inds            # ("s", "out")
y.tags            # {"output"}
```

### Example 2: TNML Model Input Preparation (NTN)

**Raw Data:**
```python
X_train = torch.tensor([
    [0.5, -0.2, 1.3, 0.8],  # 4 features
    ...
], dtype=torch.float64)  # Shape: (105, 4)
```

**Input Preparation:**
```python
loader_train = create_inputs(
    X=X_train,
    y=y_train,
    input_labels=["x0", "x1", "x2", "x3"],  # 4 input nodes (one per feature)
    output_labels=["out"],
    batch_size=32,
    append_bias=False,                      # NO bias for TNML
    encoding="polynomial",                  # Polynomial encoding
    poly_degree=3,
)

# Inside create_inputs():
# 1. Encode features:
X_encoded = encode_polynomial(X_train, degree=3)
# Shape: (105, 4, 4)  # 4 features, 4 polynomial terms each
# X_encoded[0, 0, :] = [1, 0.5, 0.25, 0.125]  # [1, x, x^2, x^3] for feature 0

# 2. Create input list:
inputs_list = [
    X_encoded[:, 0, :],  # (105, 4) - feature 0 with polynomial encoding
    X_encoded[:, 1, :],  # (105, 4) - feature 1 with polynomial encoding
    X_encoded[:, 2, :],  # (105, 4) - feature 2 with polynomial encoding
    X_encoded[:, 3, :],  # (105, 4) - feature 3 with polynomial encoding
]

# 3. Create batches:
# For batch 0 (samples 0-31):
#   input_dict = {
#       0: X_encoded[0:32, 0, :],  # (32, 4)
#       1: X_encoded[0:32, 1, :],  # (32, 4)
#       2: X_encoded[0:32, 2, :],  # (32, 4)
#       3: X_encoded[0:32, 3, :],  # (32, 4)
#   }
#
#   _prepare_batch(input_dict) creates:
#   mu = [
#       qt.Tensor(X_encoded[0:32, 0, :], inds=["s", "x0"], tags={"input_x0", "I0"}),
#       qt.Tensor(X_encoded[0:32, 1, :], inds=["s", "x1"], tags={"input_x1", "I1"}),
#       qt.Tensor(X_encoded[0:32, 2, :], inds=["s", "x2"], tags={"input_x2", "I2"}),
#       qt.Tensor(X_encoded[0:32, 3, :], inds=["s", "x3"], tags={"input_x3", "I3"}),
#   ]
```

**Batch Structure:**
```python
mu, y = loader_train[0]

# mu is a list of 4 quimb tensors (one per feature):
mu[0].data.shape  # (32, 4) - batch of 32 samples, 4 polynomial terms
mu[0].inds        # ("s", "x0")

# Each tensor contains polynomial-encoded feature:
# mu[0].data[0, :] = [1, 0.5, 0.25, 0.125]  # [1, f0, f0^2, f0^3]
```

### Example 3: GTN Input Preparation

**Raw Data:**
```python
X_train = torch.tensor([
    [0.5, -0.2, 1.3, 0.8],
    ...
], dtype=torch.float64)  # Shape: (105, 4)

y_train = torch.tensor([
    [1, 0, 0],
    ...
], dtype=torch.float64)  # Shape: (105, 3)
```

**Input Preparation (Standard Model):**
```python
# Append bias
X_train_enc = torch.cat([X_train, torch.ones(105, 1)], dim=1)
# Shape: (105, 5)

# Create PyTorch DataLoader
train_loader = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(X_train_enc, y_train),
    batch_size=32,
    shuffle=True,
)

# Iterate:
for batch_data, batch_target in train_loader:
    # batch_data.shape = (32, 5)
    # batch_target.shape = (32, 3)
    
    # In forward pass:
    output = gtn_model(batch_data)
    # gtn_model.construct_nodes(batch_data) creates:
    # [
    #     qt.Tensor(batch_data, inds=["s", "x0"], tags="Input_x0"),
    #     qt.Tensor(batch_data, inds=["s", "x1"], tags="Input_x1"),
    #     qt.Tensor(batch_data, inds=["s", "x2"], tags="Input_x2"),
    # ]
```

**Input Preparation (TNML Model):**
```python
# Encode features
X_train_enc = encode_polynomial(X_train, degree=3)
# Shape: (105, 4, 4)

# Create PyTorch DataLoader
train_loader = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(X_train_enc, y_train),
    batch_size=32,
    shuffle=True,
)

# Iterate:
for batch_data, batch_target in train_loader:
    # batch_data.shape = (32, 4, 4)  # 32 samples, 4 features, 4 polynomial terms
    # batch_target.shape = (32, 3)
    
    # In forward pass:
    prepared = prepare_input(batch_data)
    # prepared = [
    #     batch_data[:, 0, :],  # (32, 4)
    #     batch_data[:, 1, :],  # (32, 4)
    #     batch_data[:, 2, :],  # (32, 4)
    #     batch_data[:, 3, :],  # (32, 4)
    # ]
    
    output = gtn_model(prepared)
    # gtn_model.construct_nodes(prepared) creates:
    # [
    #     qt.Tensor(prepared[0], inds=["s", "x0"], tags="Input_x0"),
    #     qt.Tensor(prepared[1], inds=["s", "x1"], tags="Input_x1"),
    #     qt.Tensor(prepared[2], inds=["s", "x2"], tags="Input_x2"),
    #     qt.Tensor(prepared[3], inds=["s", "x3"], tags="Input_x3"),
    # ]
```

---

## 3. DETAILED TRAINING EXAMPLES

### Example 1: NTN Training Loop

**Setup:**
```python
# Configuration
cfg.trainer.type = "ntn"
cfg.trainer.n_epochs = 20
cfg.trainer.ridge = 5.0
cfg.trainer.ridge_decay = 0.25
cfg.trainer.ridge_min = 0.0001
cfg.trainer.patience = 10
cfg.trainer.min_delta = 0.001
cfg.trainer.train_selection = True

# Create ridge schedule
ridge_schedule = [
    max(5.0 * (0.25**epoch), 0.0001)
    for epoch in range(20)
]
# ridge_schedule = [5.0, 1.25, 0.3125, 0.078125, 0.01953125, ...]
```

**Training Loop:**
```python
# Create NTN optimizer
ntn = NTN(
    tn=model.tn,
    output_dims=["out"],
    input_dims=["x0", "x1", "x2"],
    loss=CrossEntropyLoss(),
    data_stream=loader_train,
)

# Training
metrics_log = []
best_val_quality = float("-inf")

def callback_epoch(epoch, scores_train, scores_val, info):
    metrics = {
        "epoch": epoch,
        "train_loss": float(scores_train["loss"]),
        "train_quality": float(compute_quality(scores_train)),
        "val_loss": float(scores_val["loss"]),
        "val_quality": float(compute_quality(scores_val)),
        "ridge": float(info["jitter"]),
    }
    metrics_log.append(metrics)
    print(f"Epoch {epoch}: train_loss={metrics['train_loss']:.4f}, "
          f"val_quality={metrics['val_quality']:.4f}, ridge={metrics['ridge']:.6f}")

scores_train, scores_val = ntn.fit(
    n_epochs=20,
    regularize=True,
    jitter=ridge_schedule,
    eval_metrics=CLASSIFICATION_METRICS,
    val_data=loader_val,
    verbose=True,
    callback_epoch=callback_epoch,
    adaptive_jitter=False,
    patience=10,
    min_delta=0.001,
    train_selection=True,
)

# Example output:
# Epoch 0: train_loss=1.2345, val_quality=0.4500, ridge=5.000000
# Epoch 1: train_loss=0.8234, val_quality=0.6200, ridge=1.250000
# Epoch 2: train_loss=0.5123, val_quality=0.7100, ridge=0.312500
# ...
# Epoch 15: train_loss=0.0234, val_quality=0.9200, ridge=0.000100
# Early stopping at epoch 18 (no improvement for 10 epochs)

# Extract best metrics
best_epoch = 15
best_train_loss = 0.0234
best_train_quality = 0.9300
best_val_loss = 0.0456
best_val_quality = 0.9200
```

### Example 2: GTN Training Loop

**Setup:**
```python
# Configuration
cfg.trainer.type = "gtn"
cfg.trainer.n_epochs = 1000
cfg.trainer.lr = 0.005
cfg.trainer.optimizer = "adamw"
cfg.trainer.ridge = 0.005
cfg.trainer.patience = 100
cfg.trainer.min_delta = 0.001
cfg.trainer.train_selection = True

# Create GTN model
gtn_model = GTN(
    tn=model.tn,
    output_dims=["out"],
    input_dims=["x0", "x1", "x2"],
).to(DEVICE)

# Create optimizer
weight_decay = 2 * 0.005  # 0.01
optimizer = optim.AdamW(
    gtn_model.parameters(),
    lr=0.005,
    weight_decay=0.01,
)

criterion = nn.CrossEntropyLoss()
```

**Training Loop:**
```python
metrics_log = []
best_val_quality = float("-inf")
best_epoch = -1
patience_counter = 0

for epoch in range(1000):
    # Training phase
    gtn_model.train()
    train_loss = 0.0
    
    for batch_data, batch_target in train_loader:
        batch_data, batch_target = batch_data.to(DEVICE), batch_target.to(DEVICE)
        
        # Forward pass
        optimizer.zero_grad()
        output = gtn_model(batch_data)
        loss = criterion(output, batch_target)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item() * batch_data.size(0)
    
    train_loss /= len(train_loader.dataset)
    
    # Evaluation phase
    gtn_model.eval()
    val_loss = 0.0
    all_preds, all_targets = [], []
    
    with torch.no_grad():
        for batch_data, batch_target in val_loader:
            batch_data, batch_target = batch_data.to(DEVICE), batch_target.to(DEVICE)
            output = gtn_model(batch_data)
            loss = criterion(output, batch_target)
            val_loss += loss.item() * batch_data.size(0)
            all_preds.append(output.cpu())
            all_targets.append(batch_target.cpu())
    
    val_loss /= len(val_loader.dataset)
    
    # Compute quality metrics
    preds = torch.cat(all_preds, dim=0)
    targets = torch.cat(all_targets, dim=0)
    pred_labels = preds.argmax(dim=1)
    target_labels = targets.argmax(dim=1)
    val_quality = (pred_labels == target_labels).float().mean().item()
    
    # Compute training quality
    gtn_model.eval()
    train_preds = []
    with torch.no_grad():
        for batch_data, _ in train_loader:
            batch_data = batch_data.to(DEVICE)
            output = gtn_model(batch_data)
            train_preds.append(output.cpu())
    train_preds = torch.cat(train_preds, dim=0)
    train_pred_labels = train_preds.argmax(dim=1)
    train_quality = (train_pred_labels == targets.argmax(dim=1)).float().mean().item()
    
    # Log metrics
    metrics_log.append({
        "epoch": epoch,
        "train_loss": float(train_loss),
        "train_quality": float(train_quality),
        "val_loss": float(val_loss),
        "val_quality": float(val_quality),
    })
    
    # Model selection
    if val_quality > best_val_quality + 0.001:
        best_val_quality = val_quality
        best_train_quality = train_quality
        best_epoch = epoch
        patience_counter = 0
    else:
        patience_counter += 1
    
    if epoch % 100 == 0:
        print(f"Epoch {epoch}: train_loss={train_loss:.4f}, train_quality={train_quality:.4f}, "
              f"val_loss={val_loss:.4f}, val_quality={val_quality:.4f}")
    
    # Early stopping
    if patience_counter >= 100:
        print(f"Early stopping at epoch {epoch + 1}")
        break

# Example output:
# Epoch 0: train_loss=1.0234, train_quality=0.3200, val_loss=1.0456, val_quality=0.3100
# Epoch 100: train_loss=0.2345, train_quality=0.8500, val_loss=0.2567, val_quality=0.8300
# Epoch 200: train_loss=0.0456, train_quality=0.9400, val_loss=0.0678, val_quality=0.9200
# ...
# Epoch 450: train_loss=0.0012, train_quality=0.9800, val_loss=0.0034, val_quality=0.9700
# Early stopping at epoch 551
```

---

## 4. COMMAND LINE USAGE EXAMPLES

### Basic Examples

```bash
# Default: MPO2 + iris + NTN
python run.py

# Change model
python run.py model=lmpo2
python run.py model=cpda
python run.py model=tnml_p

# Change dataset
python run.py dataset=wine
python run.py dataset=concrete
python run.py dataset=abalone

# Change trainer
python run.py trainer=gtn

# Combine
python run.py model=lmpo2 dataset=concrete trainer=gtn
```

### Hyperparameter Tuning

```bash
# Adjust bond dimension
python run.py model.bond_dim=8
python run.py model.bond_dim=10

# Adjust number of sites
python run.py model.L=4
python run.py model.L=5

# Adjust learning rate (GTN)
python run.py trainer=gtn trainer.lr=0.01
python run.py trainer=gtn trainer.lr=0.001

# Adjust ridge regularization (NTN)
python run.py trainer=ntn trainer.ridge=10
python run.py trainer=ntn trainer.ridge=1

# Adjust reduction factor (LMPO2)
python run.py model=lmpo2 model.reduction_factor=0.5
python run.py model=lmpo2 model.reduction_factor=0.7

# Adjust polynomial degree (TNML_P)
python run.py model=tnml_p model.poly_degree=2
python run.py model=tnml_p model.poly_degree=4
```

### Grid Search (Multirun)

```bash
# Sweep bond dimensions
python run.py --multirun model.bond_dim=4,6,8,10

# Sweep seeds
python run.py --multirun seed=0,1,2,3,4

# Sweep models
python run.py --multirun model=mpo2,lmpo2,mmpo2

# Sweep datasets
python run.py --multirun dataset=iris,wine,breast

# Sweep trainers
python run.py --multirun trainer=ntn,gtn

# Cartesian product
python run.py --multirun model=mpo2,lmpo2 dataset=iris,wine trainer=ntn,gtn
# Runs: 2 models × 2 datasets × 2 trainers = 8 experiments
```

### Advanced Examples

```bash
# Run with custom seed
python run.py seed=123

# Skip completed runs
python run.py skip_completed=true

# Force re-run
python run.py skip_completed=false

# Disable tracking
python run.py update_tracking=false

# Custom batch size
python run.py dataset.batch_size=64

# Multiple overrides
python run.py model=lmpo2 model.bond_dim=8 model.reduction_factor=0.5 \
              trainer=gtn trainer.lr=0.01 trainer.n_epochs=500 \
              dataset=concrete seed=42
```

---

## 5. RESULT INSPECTION

### Results File Structure

**File:** `outputs/ntn_iris_mpo2_rg5_init0.1_2024-01-15_10-30-45/results.json`

```json
{
  "success": true,
  "singular": false,
  "oom_error": false,
  "train_loss": 0.0234,
  "train_quality": 0.9300,
  "val_loss": 0.0456,
  "val_quality": 0.9200,
  "best_epoch": 15,
  "metrics_log": [
    {
      "epoch": 0,
      "train_loss": 1.2345,
      "train_quality": 0.4500,
      "val_loss": 1.3456,
      "val_quality": 0.4200,
      "ridge": 5.0
    },
    {
      "epoch": 1,
      "train_loss": 0.8234,
      "train_quality": 0.6200,
      "val_loss": 0.9123,
      "val_quality": 0.6000,
      "ridge": 1.25
    },
    ...
  ],
  "config": {
    "model": {
      "name": "MPO2",
      "L": 3,
      "bond_dim": 6,
      "output_site": null,
      "init_strength": 0.1
    },
    "trainer": {
      "type": "ntn",
      "n_epochs": 20,
      "ridge": 5,
      "ridge_decay": 0.25,
      "ridge_min": 0.0001,
      "patience": 10,
      "min_delta": 0.001,
      "train_selection": true
    },
    "dataset": {
      "name": "iris",
      "task": "classification",
      "batch_size": 32
    },
    "seed": 42
  },
  "dataset_info": {
    "name": "iris",
    "dataset_id": 53,
    "n_samples": 150,
    "n_train": 105,
    "n_val": 22,
    "n_test": 23,
    "n_features": 4,
    "task": "classification",
    "n_classes": 3
  }
}
```

### Analyzing Results

```python
import json
import pandas as pd

# Load results
with open("results.json") as f:
    result = json.load(f)

# Check success
print(f"Success: {result['success']}")
print(f"Best epoch: {result['best_epoch']}")
print(f"Val quality: {result['val_quality']:.4f}")

# Plot training curve
metrics_df = pd.DataFrame(result['metrics_log'])
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(metrics_df['epoch'], metrics_df['train_loss'], label='Train')
plt.plot(metrics_df['epoch'], metrics_df['val_loss'], label='Val')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Loss Curve')

plt.subplot(1, 2, 2)
plt.plot(metrics_df['epoch'], metrics_df['train_quality'], label='Train')
plt.plot(metrics_df['epoch'], metrics_df['val_quality'], label='Val')
plt.xlabel('Epoch')
plt.ylabel('Quality')
plt.legend()
plt.title('Quality Curve')

plt.tight_layout()
plt.savefig('training_curve.png')
```

