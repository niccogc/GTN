# type: ignore
"""
Test LMPO2 on Iris dataset
"""
import torch
import numpy as np
from sklearn import datasets
from model.MPO2_models import LMPO2
from model.NTN import NTN
from model.builder import Inputs
from model.losses import CrossEntropyLoss
from model.utils import CLASSIFICATION_METRICS

torch.set_default_dtype(torch.float32)

print("="*70)
print("TESTING LMPO2 ON IRIS DATASET")
print("="*70)

# Load Iris
iris = datasets.load_iris()
X = torch.tensor(iris.data, dtype=torch.float32)
y = torch.tensor(iris.target, dtype=torch.long)

print(f"\nData: X shape={X.shape}, y shape={y.shape}")
print(f"Classes: {np.bincount(y.numpy())}")

# Normalize and pad
X = (X - X.mean(dim=0)) / X.std(dim=0)
X_padded = torch.cat([X, torch.ones(X.shape[0], 1)], dim=1)  # (150, 5)

print(f"Padded X shape: {X_padded.shape}")

# Configuration
N_CLASSES = 3
L = 3
BOND_DIM = 8
INPUT_DIM = 5
REDUCTION_FACTOR = 0.5  # 50% reduction
REDUCED_DIM = max(2, int(INPUT_DIM * REDUCTION_FACTOR))  # At least 2
BATCH_SIZE = 30
N_EPOCHS = 15

print(f"\nConfiguration:")
print(f"  Sites (L): {L}")
print(f"  Bond dimension: {BOND_DIM}")
print(f"  Input dimension: {INPUT_DIM}")
print(f"  Reduction factor: {REDUCTION_FACTOR*100:.0f}%")
print(f"  Reduced dimension: {REDUCED_DIM} ({REDUCED_DIM/INPUT_DIM*100:.0f}% of input)")
print(f"  Output classes: {N_CLASSES}")
print(f"  Batch size: {BATCH_SIZE}")
print(f"  Epochs: {N_EPOCHS}")

# Create LMPO2 using reduction_factor
lmpo2 = LMPO2(
    L=L,
    bond_dim=BOND_DIM,
    input_dim=INPUT_DIM,
    reduction_factor=REDUCTION_FACTOR,
    output_dim=N_CLASSES
)

print(f"\nActual reduced dimension: {lmpo2.reduced_dim}")

print(f"\nLMPO2 Structure:")
print(f"  Total tensors: {len(lmpo2.tn.tensors)}")
print(f"  MPO tensors: {len(lmpo2.mpo_tensors)}")
print(f"  MPS tensors: {len(lmpo2.mps_tensors)}")
print(f"  Trainable nodes: {[list(t.tags) for t in lmpo2.tn.tensors]}")

# Setup data loader
loader = Inputs(
    inputs=[X_padded],
    outputs=[y.unsqueeze(1)],
    outputs_labels=lmpo2.output_dims,
    input_labels=lmpo2.input_labels,
    batch_dim="s",
    batch_size=BATCH_SIZE
)

# Create NTN model
loss = CrossEntropyLoss()
model = NTN(
    tn=lmpo2.tn,
    output_dims=lmpo2.output_dims,
    input_dims=lmpo2.input_dims,
    loss=loss,
    data_stream=loader
)

# Jitter schedule: slower decay
jitter_schedule = [max(5.0 * (0.8 ** epoch), 0.1) for epoch in range(N_EPOCHS)]
print(f"\nJitter schedule: {jitter_schedule[0]:.3f} -> {jitter_schedule[-1]:.3f}")

print(f"\n{'='*70}")
print("TRAINING")
print(f"{'='*70}")

try:
    final_metrics = model.fit(
        n_epochs=N_EPOCHS,
        regularize=True,
        jitter=jitter_schedule,
        eval_metrics=CLASSIFICATION_METRICS,
        verbose=True
    )
    
    print(f"\n{'='*70}")
    print("RESULTS")
    print(f"{'='*70}")
    print(f"Final Loss: {final_metrics['loss']:.4f}")
    print(f"Final Accuracy: {final_metrics['accuracy']:.2%}")
    
    import math
    if math.isnan(final_metrics['loss']):
        print("\n✗ FAILED: Loss is NaN")
    elif final_metrics['accuracy'] > 0.9:
        print("\n✓ EXCELLENT: >90% accuracy")
    elif final_metrics['accuracy'] > 0.7:
        print("\n✓ GOOD: >70% accuracy")
    elif final_metrics['accuracy'] > 0.5:
        print("\n✓ OK: Above random baseline (33%)")
    else:
        print("\n✗ POOR: Close to random baseline")
        
except Exception as e:
    print(f"\n✗ ERROR during training:")
    print(f"  {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
