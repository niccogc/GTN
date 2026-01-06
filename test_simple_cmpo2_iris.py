# type: ignore
"""
Test SimpleCMPO2_NTN (no caching) on Iris dataset
"""
import torch
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
import quimb.tensor as qt
from model.MPS_simple import SimpleCMPO2_NTN
from model.builder import Inputs
from model.losses import CrossEntropyLoss
from model.utils import CLASSIFICATION_METRICS

torch.set_default_dtype(torch.float32)

print("="*70)
print("SIMPLE CMPO2_NTN ON IRIS DATASET (NO CACHING)")
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

# For CMPO2, we need (n_samples, n_patches, n_pixels)
# Let's reshape: 4 features -> 2 patches x 2 pixels
n_samples = X.shape[0]
n_patches = 2
n_pixels = 2

X_reshaped = X.reshape(n_samples, n_patches, n_pixels)

print(f"  Reshaped X: {X_reshaped.shape} (samples, patches, pixels)")

# Configuration
N_CLASSES = 3
L = 2  # 2 sites
BOND_DIM = 8  # Increase bond dimension
BATCH_SIZE = 30
N_EPOCHS = 50  # More epochs

print(f"\nConfiguration:")
print(f"  Sites (L): {L}")
print(f"  Bond dimension: {BOND_DIM}")
print(f"  Batch size: {BATCH_SIZE}")
print(f"  Total samples: {len(X_reshaped)}")
print(f"  Training epochs: {N_EPOCHS}")
print(f"  Number of classes: {N_CLASSES}")
print(f"  Patches: {n_patches}, Pixels: {n_pixels}")

# Setup MPS with output dimension = N_CLASSES
psi = qt.MPS_rand_state(L, bond_dim=BOND_DIM, phys_dim=n_pixels)
phi = qt.MPS_rand_state(L, bond_dim=BOND_DIM, phys_dim=n_patches)

# Add output dimension to middle node (last node for L=2)
middle_psi = psi['I1']
middle_psi.new_ind('out', size=N_CLASSES, axis=-1, mode='random', rand_strength=0.01)
psi.apply_to_arrays(lambda x: torch.tensor(x, dtype=torch.float32))
phi.apply_to_arrays(lambda x: torch.tensor(x, dtype=torch.float32))

# Reindex physical dimensions
psi.reindex({f"k{i}": f"{i}_pixels" for i in range(L)}, inplace=True)
phi.reindex({f"k{i}": f"{i}_patches" for i in range(L)}, inplace=True)

# Add tags
for i in range(L):
    psi.add_tag(f"{i}_Pi", where=f"I{i}")
    phi.add_tag(f"{i}_Pa", where=f"I{i}")

tn = psi & phi

print(f"\nTensor Network:")
print(f"  Total tensors: {len(tn.tensors)}")
print(f"  Outer indices: {tn.outer_inds()}")

# Setup data loader
# Input labels format: [source_idx, (patch_ind, pixel_ind)]
input_labels_cmpo2 = [
    [0, (f"{i}_patches", f"{i}_pixels")]
    for i in range(L)
]

# Simple labels for model.input_dims
input_labels = [str(i) for i in range(L)]

loader = Inputs(
    inputs=[X_reshaped],
    outputs=[y.unsqueeze(1)],  # NTN expects (N, 1) for class indices
    outputs_labels=["out"],
    input_labels=input_labels_cmpo2,
    batch_dim="s",
    batch_size=BATCH_SIZE
)

# Setup loss
loss = CrossEntropyLoss()

# Create simple model (NO CACHING)
model = SimpleCMPO2_NTN(
    tn=tn,
    output_dims=["out"],
    input_dims=input_labels,
    loss=loss,
    data_stream=loader,
    psi=psi,
    phi=phi
)

print(f"\nModel:")
print(f"  Type: SimpleCMPO2_NTN (no caching)")
print(f"  Loss: CrossEntropyLoss")

print(f"\n" + "-"*70)
print("TRAINING")
print("-"*70)

# Jitter schedule: start from 5.0, multiply by 0.1 every 10 epochs
jitter_schedule = [5.0 * (0.1 ** (epoch // 10)) for epoch in range(N_EPOCHS)]

print(f"Jitter schedule (first 10): {[f'{j:.6e}' for j in jitter_schedule[:10]]}")

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
    elif final_metrics['accuracy'] > 0.5:  # Random baseline for 3 classes is 33%
        print(f"\n✓ SUCCESS: Accuracy {final_metrics['accuracy']:.2%} is well above random baseline (33%)")
    elif final_metrics['accuracy'] > 0.4:
        print(f"\n✓ SUCCESS: Accuracy {final_metrics['accuracy']:.2%} is above random baseline (33%)")
    else:
        print(f"\n✗ WARNING: Accuracy {final_metrics['accuracy']:.2%} is close to random baseline (33%)")

except Exception as e:
    print(f"\n✗ ERROR during training:")
    print(f"  {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
