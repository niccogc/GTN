# type: ignore
"""
Test MPO2 (simple MPS with output dimension) on Iris dataset
"""
import torch
import numpy as np
from sklearn import datasets
import quimb.tensor as qt
from model.NTN import NTN
from model.losses import MSELoss
from model.utils import CLASSIFICATION_METRICS, create_inputs
from model.MPO2_models import MPO2

torch.set_default_dtype(torch.float32)

print("="*70)
print("MPO2 (SIMPLE MPS) ON IRIS DATASET")
print("="*70)

# Load Iris dataset
print("\n1. Loading Iris dataset...")
iris = datasets.load_iris()
X = iris.data  # (150, 4) - 4 features
y = iris.target  # (150,) - 3 classes

print(f"  Dataset: {len(X)} samples, {X.shape[1]} features, {len(np.unique(y))} classes")
print(f"  Class distribution: {np.bincount(y)}")

# Convert to torch
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.long)

# Normalize features
X = (X - X.mean(dim=0)) / X.std(dim=0)

# Pad to 5 features (phys_dim=5)
n_samples = X.shape[0]
X_features = torch.cat([X, torch.ones(n_samples, 1)], dim=1)  # (150, 5)

print(f"  X shape: {X_features.shape} (samples, features)")

# Configuration
N_CLASSES = 3
L = 3  # 3 sites
PHYS_DIM = 5  # physical dimension 5
BOND_DIM = 8
BATCH_SIZE = 30
N_EPOCHS = 20

print(f"\nConfiguration:")
print(f"  Sites (L): {L}")
print(f"  Physical dimension: {PHYS_DIM}")
print(f"  Bond dimension: {BOND_DIM}")
print(f"  Batch size: {BATCH_SIZE}")
print(f"  Total samples: {len(X_features)}")
print(f"  Training epochs: {N_EPOCHS}")
print(f"  Number of classes: {N_CLASSES}")

# Convert y to one-hot for MSE loss
import torch.nn.functional as F
y_onehot = F.one_hot(y, num_classes=N_CLASSES).float()

# Create MPO2 model
print("\n2. Creating MPO2 model...")
mpo2 = MPO2(
    L=L,
    bond_dim=BOND_DIM,
    phys_dim=PHYS_DIM,
    output_dim=N_CLASSES,
    output_site=1,  # Middle site
    init_strength=0.1
)

print(f"  MPO2 created with:")
print(f"    L={L}, bond_dim={BOND_DIM}, phys_dim={PHYS_DIM}, output_dim={N_CLASSES}")
print(f"    Output site: {mpo2.output_site}")
print(f"    Input labels: {mpo2.input_labels}")
print(f"    Input dims: {mpo2.input_dims}")
print(f"    Output dims: {mpo2.output_dims}")

# Setup data loader using utility function
loader = create_inputs(
    X=X_features,
    y=y_onehot,
    input_labels=mpo2.input_labels,
    output_labels=mpo2.output_dims,
    batch_size=BATCH_SIZE,
    batch_dim="s"
)

# Setup loss
loss = MSELoss()

# Create NTN model
model = NTN(
    tn=mpo2.tn,
    output_dims=mpo2.output_dims,
    input_dims=mpo2.input_dims,
    loss=loss,
    data_stream=loader
)

print(f"\nModel:")
print(f"  Type: NTN with MPO2")
print(f"  Loss: MSELoss")
print(f"  Total tensors: {len(model.tn.tensors)}")
print(f"  Outer indices: {model.tn.outer_inds()}")

print(f"\n" + "-"*70)
print("TRAINING")
print("-"*70)

# Jitter schedule: start from 5.0 and multiply by 0.1 each epoch
jitter_schedule = [5.0 * (0.1 ** epoch) for epoch in range(N_EPOCHS)]

print(f"Jitter schedule: start=5.0, decay=0.1x per epoch")

try:
    final_metrics = model.fit(
        n_epochs=N_EPOCHS,
        regularize=True,
        jitter=jitter_schedule,
        eval_metrics=CLASSIFICATION_METRICS,
        verbose=True
    )
    
    print(f"\n" + "-"*70)
    print("TRAINING COMPLETE")
    print("-"*70)
    
    # Print final metrics
    print(f"\nFinal Metrics (Epoch {N_EPOCHS}):")
    print(f"  Loss: {final_metrics['loss']:.4f}")
    print(f"  Accuracy: {final_metrics['accuracy']:.2%}")
    
    # Check if loss is valid
    import math
    if math.isnan(final_metrics['loss']):
        print(f"\n✗ ERROR: Loss is NaN - training unstable!")
    elif final_metrics['accuracy'] > 0.9:
        print(f"\n✓ SUCCESS: Accuracy {final_metrics['accuracy']:.2%} - excellent!")
    elif final_metrics['accuracy'] > 0.5:
        print(f"\n✓ SUCCESS: Accuracy {final_metrics['accuracy']:.2%} is well above random baseline (33%)")
    else:
        print(f"\n✗ WARNING: Accuracy {final_metrics['accuracy']:.2%} is close to random baseline (33%)")

except Exception as e:
    print(f"\n✗ ERROR during training:")
    print(f"  {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
