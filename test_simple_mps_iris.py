# type: ignore
"""
Test simple MPS (not CMPO2) on Iris dataset
"""
import torch
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
import quimb.tensor as qt
from model.NTN import NTN
from model.builder import Inputs
from model.losses import CrossEntropyLoss, MSELoss
from model.utils import CLASSIFICATION_METRICS

torch.set_default_dtype(torch.float32)

print("="*70)
print("SIMPLE MPS ON IRIS DATASET")
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

# Setup data loader first
# The builder will automatically repeat the input for each site
# Input labels: just the index names
input_labels = ["x0", "x1", "x2"]

loader = Inputs(
    inputs=[X_features],  # Shape: (150, 5) - builder repeats this for each site
    outputs=[y_onehot],  # Use one-hot for MSE loss (150, 3)
    outputs_labels=["out"],
    input_labels=input_labels,
    batch_dim="s",
    batch_size=BATCH_SIZE
)

# Create MPS manually (like in test.py)
def init_weights(shape):
    w = torch.randn(*shape) * 0.1
    return w

t0 = qt.Tensor(data=init_weights((PHYS_DIM, BOND_DIM)), inds=('x0', 'b0'), tags={'Node0'})
t1 = qt.Tensor(data=init_weights((BOND_DIM, PHYS_DIM, BOND_DIM, N_CLASSES)), inds=('b0', 'x1', 'b1', 'out'), tags={'Node1'})
t2 = qt.Tensor(data=init_weights((BOND_DIM, PHYS_DIM)), inds=('b1', 'x2'), tags={'Node2'})
tn = qt.TensorNetwork([t0, t1, t2])

print(f"\nMPS:")
print(f"  Total tensors: {len(tn.tensors)}")
print(f"  Outer indices: {tn.outer_inds()}")

# Setup loss - use MSE for debugging
loss = MSELoss()

# Create model
model = NTN(
    tn=tn,
    output_dims=["out"],
    input_dims=input_labels,
    loss=loss,
    data_stream=loader
)

print(f"\nModel:")
print(f"  Type: NTN (simple MPS)")
print(f"  Loss: CrossEntropyLoss")

print(f"\n" + "-"*70)
print("TRAINING")
print("-"*70)

# Jitter schedule: start from 5.0 and multiply by 0.1 each epoch (no minimum)
jitter_schedule = [5.0 * (0.1 ** epoch) for epoch in range(N_EPOCHS)]

print(f"Jitter schedule: {[f'{j:.6e}' for j in jitter_schedule]}")

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
