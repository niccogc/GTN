# GTN Architecture - Detailed Examples

## Example 1: Simple MPO2 Regression

### Setup

```python
import torch
from model.standard.MPO2_models import MPO2
from model.utils import create_inputs
from model.losses import MSELoss
from model.base.NTN import NTN

# Create synthetic data
X = torch.randn(100, 3)  # 100 samples, 3 features
y = torch.randn(100, 1)  # 100 targets

# Create model
model = MPO2(
    L=3,                    # 3 sites (one per feature)
    bond_dim=4,             # Bond dimension
    phys_dim=2,             # Physical dimension per site
    output_dim=1,           # Single output
    output_site=2,          # Output at last site
    use_tn_normalization=True,
    tn_target_std=0.1
)

# Prepare inputs with bias
inputs = create_inputs(
    X, y,
    input_labels=["x"],     # Single input tensor
    output_labels=["out"],
    batch_size=32,
    append_bias=True        # Adds column of ones
)

# Create loss and trainer
loss = MSELoss()
trainer = NTN(
    tn=model.tn,
    output_dims=model.output_dims,
    input_dims=model.input_dims,
    loss=loss,
    data_stream=inputs
)

# Train
train_scores, val_scores = trainer.fit(
    n_epochs=10,
    regularize=True,
    jitter=1e-6,
    verbose=True
)
```

### Tensor Shapes During Training

```
Input tensor:
  Shape: (batch=32, features=4)  # 3 features + 1 bias
  Indices: ("s", "x")
  Tags: {"input_x", "I0"}

Network tensors:
  Node0: (2, 4)     indices: ("x0", "b0")
  Node1: (4, 2, 4)  indices: ("b0", "x1", "b1")
  Node2: (4, 2, 1)  indices: ("b1", "x2", "out")

Forward pass:
  full_tn = model.tn & [input_tensor]
  y_pred = full_tn.contract(output_inds=["s", "out"])
  y_pred shape: (32, 1)

Environment for Node2:
  env = contract(model.tn \ {Node2} & input, output_inds=["s", "b1", "x2", "out"])
  env shape: (32, 4, 2, 1)

Gradient computation:
  dL_dy shape: (32, 1)
  node_grad = contract(env & dL_dy, output_inds=["b1", "x2", "out"])
  node_grad shape: (4, 2, 1)

Hessian computation:
  d2L_dy2 shape: (32, 1, 1)
  node_hess = contract(env & d2L_dy2 & env_prime, 
                       output_inds=["b1", "x2", "out", "b1_prime", "x2_prime", "out_prime"])
  node_hess shape: (4, 2, 1, 4, 2, 1)
```

---

## Example 2: CMPO2 with Cross-Connection

### Setup

```python
from model.standard.MPO2_models import CMPO2

# Create model with two input types
model = CMPO2(
    L=4,                      # 4 sites
    bond_dim=3,               # Bond dimension
    phys_dim_pixels=5,        # Pixel features per site
    phys_dim_patches=3,       # Patch features per site
    output_dim=2,             # Binary classification
    output_site=3
)

# Input preparation
X_patches = torch.randn(100, 4, 3)  # 100 samples, 4 sites, 3 patch features
X_pixels = torch.randn(100, 4, 5)   # 100 samples, 4 sites, 5 pixel features

# Flatten for input
X_patches_flat = X_patches.reshape(100, -1)  # (100, 12)
X_pixels_flat = X_pixels.reshape(100, -1)    # (100, 20)

inputs = Inputs(
    inputs=[X_patches_flat, X_pixels_flat],
    outputs=[y],
    input_labels=[[0, ("0_patches", "0_pixels")], 
                  [0, ("1_patches", "1_pixels")],
                  [0, ("2_patches", "2_pixels")],
                  [0, ("3_patches", "3_pixels")]],
    batch_dim="s",
    batch_size=32
)
```

### Index Structure

```
Input tensor 0:
  Shape: (batch, 12)
  Indices: ("s", "0_patches", "0_pixels")
  Tags: {"input_0_patches_0_pixels", "I0"}

Network structure:
  Pixel MPS (psi):
    Node0_Pi: (5, 3)      indices: ("0_pixels", "b_psi_0")
    Node1_Pi: (3, 5, 3)   indices: ("b_psi_0", "1_pixels", "b_psi_1")
    Node2_Pi: (3, 5, 3)   indices: ("b_psi_1", "2_pixels", "b_psi_2")
    Node3_Pi: (3, 5, 2)   indices: ("b_psi_2", "3_pixels", "out")
  
  Patch MPS (phi):
    Node0_Pa: (3, 3)      indices: ("0_patches", "b_phi_0")
    Node1_Pa: (3, 3, 3)   indices: ("b_phi_0", "1_patches", "b_phi_1")
    Node2_Pa: (3, 3, 3)   indices: ("b_phi_1", "2_patches", "b_phi_2")
    Node3_Pa: (3, 3, 3)   indices: ("b_phi_2", "3_patches", "b_phi_3")
```

---

## Example 3: LMPO2 with Dimensionality Reduction

### Setup

```python
from model.standard.MPO2_models import LMPO2

# Create model with reduction
model = LMPO2(
    L=5,                      # 5 sites
    bond_dim=4,               # MPS bond dimension
    phys_dim=10,              # Input dimension per site
    reduced_dim=5,            # Reduced dimension (50% reduction)
    output_dim=3,             # 3-class classification
    output_site=4,
    bond_dim_mpo=2,           # MPO bond dimension
    use_tn_normalization=True
)

# Input: 5 features, each with dimension 10
inputs = create_inputs(
    X,  # (100, 5)
    y,
    input_labels=[f"{i}_in" for i in range(5)],
    batch_size=32
)
```

### Two-Layer Structure

```
Layer 1: MPO (Dimensionality Reduction)
  Site 0: (10, 5, 2)      indices: ("0_in", "0_reduced", "b_mpo_0")
  Site 1: (2, 10, 5, 2)   indices: ("b_mpo_0", "1_in", "1_reduced", "b_mpo_1")
  Site 2: (2, 10, 5, 2)   indices: ("b_mpo_1", "2_in", "2_reduced", "b_mpo_2")
  Site 3: (2, 10, 5, 2)   indices: ("b_mpo_2", "3_in", "3_reduced", "b_mpo_3")
  Site 4: (2, 10, 5)      indices: ("b_mpo_3", "4_in", "4_reduced")

Layer 2: MPS (Output)
  Site 0: (5, 4)          indices: ("0_reduced", "b_mps_0")
  Site 1: (4, 5, 4)       indices: ("b_mps_0", "1_reduced", "b_mps_1")
  Site 2: (4, 5, 4)       indices: ("b_mps_1", "2_reduced", "b_mps_2")
  Site 3: (4, 5, 4)       indices: ("b_mps_2", "3_reduced", "b_mps_3")
  Site 4: (4, 5, 3)       indices: ("b_mps_3", "4_reduced", "out")

Forward pass:
  Input (batch, 5) → MPO → (batch, 5) → MPS → (batch, 3)
```

---

## Example 4: MMPO2 with Non-Trainable Mask

### Setup

```python
from model.standard.MPO2_models import MMPO2

# Create model with causal mask
model = MMPO2(
    L=3,
    bond_dim=4,
    phys_dim=2,
    output_dim=1,
    output_site=2,
    use_tn_normalization=True
)

# The mask is non-trainable (marked with "NT" tag)
# Only the MPS layer is trainable
```

### Mask Structure

```
Heaviside Matrix H (causal mask):
  H[i, j] = 1 if j >= i else 0
  
  H = [[1, 1, 1],
       [0, 1, 1],
       [0, 0, 1]]

Mask Tensors (Non-Trainable, "NT" tag):
  Site 0: (2, 2, 2)      indices: ("0_in", "0_masked", "b_mask_0")
          Tags: {"0_Mask", "NT"}
  
  Site 1: (2, 2, 2, 2)   indices: ("b_mask_0", "1_in", "1_masked", "b_mask_1")
          Tags: {"1_Mask", "NT"}
  
  Site 2: (2, 2, 2)      indices: ("b_mask_1", "2_in", "2_masked")
          Tags: {"2_Mask", "NT"}

MPS Tensors (Trainable):
  Site 0: (2, 4)         indices: ("0_masked", "b_mps_0")
          Tags: {"0_MPS"}
  
  Site 1: (4, 2, 4)      indices: ("b_mps_0", "1_masked", "b_mps_1")
          Tags: {"1_MPS"}
  
  Site 2: (4, 2, 1)      indices: ("b_mps_1", "2_masked", "out")
          Tags: {"2_MPS"}
```

---

## Example 5: Classification with CrossEntropyLoss

### Setup

```python
from model.losses import CrossEntropyLoss

# Create model for 4-class classification
model = MPO2(
    L=5,
    bond_dim=4,
    phys_dim=3,
    output_dim=4,           # 4 classes
    use_tn_normalization=True
)

# Prepare data
X = torch.randn(200, 5)
y = torch.randint(0, 4, (200,))  # Class indices

inputs = create_inputs(X, y, input_labels=["x"], batch_size=32)

# Create loss (full Hessian required!)
loss = CrossEntropyLoss(use_diagonal_hessian=False)

trainer = NTN(
    tn=model.tn,
    output_dims=model.output_dims,
    input_dims=model.input_dims,
    loss=loss,
    data_stream=inputs
)

# Train
trainer.fit(n_epochs=20, regularize=True, jitter=1e-5)
```

### Gradient/Hessian Computation

```
Forward pass:
  y_pred shape: (batch, 4)  # Logits for 4 classes

Loss derivatives:
  Gradient: dL/dz_i = p_i - y_i  (p = softmax(logits))
  dL_dy shape: (batch, 4)
  
  Hessian: d²L/dz_i dz_j = p_i * (δ_ij - p_j)  (FULL MATRIX!)
  d2L_dy2 shape: (batch, 4, 4)  # Full matrix, not diagonal!

Node update:
  Hessian is (batch, 4, 4) → fuse to (batch, 16, 16)
  Gradient is (batch, 4) → fuse to (batch, 16)
  Solve: H * Δw = -∇L
```

---

## Example 6: Polynomial Features

### Setup

```python
# Create data
X = torch.randn(100, 3)
y = torch.randn(100, 1)

# Create inputs with polynomial encoding
inputs = create_inputs(
    X, y,
    input_labels=["x"],
    encoding="polynomial",
    poly_degree=2,
    batch_size=32
)

# Model expects phys_dim = degree + 1 = 3
model = MPO2(
    L=3,
    bond_dim=4,
    phys_dim=3,             # [1, x_i, x_i^2]
    output_dim=1,
    use_tn_normalization=True
)
```

### Feature Transformation

```
Input X: (100, 3)
  X = [[x_0^(0), x_1^(0), x_2^(0)],
       [x_0^(1), x_1^(1), x_2^(1)],
       ...]

Polynomial encoding (degree=2):
  X_encoded: (100, 3, 3)
  X_encoded = [[[1, x_0, x_0^2],
                [1, x_1, x_1^2],
                [1, x_2, x_2^2]],
               ...]

Input tensors:
  Shape: (batch, 3)
  Indices: ("s", "x")
  
  For each batch sample:
    [1, x_0, x_0^2] → connects to Node0
    [1, x_1, x_1^2] → connects to Node1
    [1, x_2, x_2^2] → connects to Node2
```

---

## Example 7: Fourier Features

### Setup

```python
# Create inputs with Fourier encoding
inputs = create_inputs(
    X, y,
    input_labels=["x"],
    encoding="fourier",
    batch_size=32
)

# Model expects phys_dim = 2
model = MPO2(
    L=3,
    bond_dim=4,
    phys_dim=2,             # [cos(π*x_i/2), sin(π*x_i/2)]
    output_dim=1,
    use_tn_normalization=True
)
```

### Feature Transformation

```
Input X: (100, 3)

Fourier encoding:
  X_encoded: (100, 3, 2)
  X_encoded = [[[cos(π*x_0/2), sin(π*x_0/2)],
                [cos(π*x_1/2), sin(π*x_1/2)],
                [cos(π*x_2/2), sin(π*x_2/2)]],
               ...]

Input tensors:
  Shape: (batch, 2)
  Indices: ("s", "x")
```

---

## Example 8: Early Stopping with Validation

### Setup

```python
# Split data
n_train = 80
n_val = 20

X_train = X[:n_train]
y_train = y[:n_train]
X_val = X[n_train:]
y_val = y[n_train:]

# Create input streams
train_inputs = create_inputs(X_train, y_train, batch_size=16)
val_inputs = create_inputs(X_val, y_val, batch_size=16)

# Train with early stopping
train_scores, val_scores = trainer.fit(
    n_epochs=100,
    regularize=True,
    jitter=1e-6,
    val_data=val_inputs,
    patience=10,            # Stop if no improvement for 10 epochs
    min_delta=0.001,        # Minimum improvement threshold
    verbose=True
)
```

### Output

```
Init    | Train: loss: 0.50000 | quality: 0.00000 | Val:   loss: 0.51000 | quality: -0.01000
Epoch 1 | Train: loss: 0.45000 | quality: 0.10000 | Val:   loss: 0.46000 | quality: 0.09000 *
Epoch 2 | Train: loss: 0.42000 | quality: 0.15000 | Val:   loss: 0.44000 | quality: 0.12000 *
...
Epoch 15 | Train: loss: 0.30000 | quality: 0.40000 | Val:   loss: 0.35000 | quality: 0.35000
Epoch 16 | Train: loss: 0.29000 | quality: 0.41000 | Val:   loss: 0.35500 | quality: 0.34500
...
Epoch 25 | Train: loss: 0.28000 | quality: 0.42000 | Val:   loss: 0.36000 | quality: 0.34000

⏸ Early stopping at epoch 26 (best was epoch 15)
```

---

## Example 9: Custom Regularization Schedule

### Setup

```python
# Jitter schedule: decrease over epochs
n_epochs = 50
jitter_schedule = [1e-4 * (0.9 ** i) for i in range(n_epochs)]

trainer.fit(
    n_epochs=n_epochs,
    regularize=True,
    jitter=jitter_schedule,  # List of jitter values
    verbose=True
)
```

### Effect

```
Epoch 1:  jitter = 1.0e-4
Epoch 2:  jitter = 9.0e-5
Epoch 3:  jitter = 8.1e-5
...
Epoch 50: jitter = 2.6e-6

This gradually reduces regularization as training progresses,
allowing the model to fit more precisely in later epochs.
```

---

## Example 10: Debugging Tensor Shapes

### Inspection Code

```python
# Inspect model structure
print("Model Tensors:")
for i, tensor in enumerate(model.tn):
    print(f"  Tensor {i}: shape={tensor.shape}, inds={tensor.inds}, tags={tensor.tags}")

# Inspect input structure
print("\nInput Structure:")
mu, y = next(iter(inputs.data_mu_y))
for i, input_tensor in enumerate(mu):
    print(f"  Input {i}: shape={input_tensor.shape}, inds={input_tensor.inds}, tags={input_tensor.tags}")
print(f"  Output: shape={y.shape}, inds={y.inds}, tags={y.tags}")

# Inspect environment
env = trainer.get_environment(
    trainer.tn, 
    "Node2", 
    inputs.data_mu,
    sum_over_batch=False,
    sum_over_output=False
)
print(f"\nEnvironment for Node2: shape={env.shape}, inds={env.inds}")

# Inspect gradient/hessian
J, H = trainer._compute_H_b("Node2")
print(f"\nGradient: shape={J.shape}, inds={J.inds}")
print(f"Hessian: shape={H.shape}, inds={H.inds}")
```

### Output

```
Model Tensors:
  Tensor 0: shape=(2, 4), inds=('x0', 'b0'), tags={'Node0'}
  Tensor 1: shape=(4, 2, 4), inds=('b0', 'x1', 'b1'), tags={'Node1'}
  Tensor 2: shape=(4, 2, 1), inds=('b1', 'x2', 'out'), tags={'Node2'}

Input Structure:
  Input 0: shape=(32, 4), inds=('s', 'x'), tags={'input_x', 'I0'}
  Output: shape=(32, 1), inds=('s', 'out'), tags={'output'}

Environment for Node2: shape=(32, 4, 2, 1), inds=('s', 'b1', 'x2', 'out')

Gradient: shape=(4, 2, 1), inds=('b1', 'x2', 'out')
Hessian: shape=(4, 2, 1, 4, 2, 1), inds=('b1', 'x2', 'out', 'b1_prime', 'x2_prime', 'out_prime')
```

