# type: ignore
"""
Test LMPO2 and MMPO2 on Iris dataset
(CMPO2 is excluded because it requires patches×pixels format)
"""
import torch
import numpy as np
from sklearn import datasets
from model.MPO2_models import LMPO2, MMPO2
from model.NTN import NTN
from model.builder import Inputs
from model.losses import CrossEntropyLoss
from model.utils import CLASSIFICATION_METRICS

torch.set_default_dtype(torch.float32)

print("="*70)
print("TESTING LMPO2 AND MMPO2 ON IRIS DATASET")
print("="*70)

# Load Iris dataset
print("\n1. Loading Iris dataset...")
iris = datasets.load_iris()
X = iris.data  # (150, 4)
y = iris.target  # (150,)

print(f"  Dataset: {len(X)} samples, {X.shape[1]} features, {len(np.unique(y))} classes")

# Convert to torch and normalize
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.long)
X = (X - X.mean(dim=0)) / X.std(dim=0)

# Pad to 5 features
X_padded = torch.cat([X, torch.ones(X.shape[0], 1)], dim=1)  # (150, 5)
print(f"  X shape: {X_padded.shape}")

# Configuration
N_CLASSES = 3
L = 3
BOND_DIM = 8
BATCH_SIZE = 30
N_EPOCHS = 20

print(f"\nConfiguration:")
print(f"  Sites (L): {L}")
print(f"  Bond dimension: {BOND_DIM}")
print(f"  Batch size: {BATCH_SIZE}")
print(f"  Epochs: {N_EPOCHS}")
print(f"  Classes: {N_CLASSES}")

# Jitter schedule: start at 5.0, multiply by 0.5 each epoch, min 1e-3
jitter_schedule = [max(5.0 * (0.5 ** epoch), 1e-3) for epoch in range(N_EPOCHS)]
print(f"  Jitter: {jitter_schedule[0]:.3f} -> {jitter_schedule[-1]:.6f}")

def test_model(model_name, model_obj):
    """Test a single model."""
    print(f"\n{'='*70}")
    print(f"TESTING {model_name}")
    print(f"{'='*70}")
    
    print(f"\nTensor Network:")
    print(f"  Total tensors: {len(model_obj.tn.tensors)}")
    print(f"  Outer indices: {model_obj.tn.outer_inds()}")
    trainable = [t.tags for t in model_obj.tn.tensors if 'NOT_TRAINABLE' not in t.tags]
    not_trainable = [t.tags for t in model_obj.tn.tensors if 'NOT_TRAINABLE' in t.tags]
    print(f"  Trainable nodes ({len(trainable)}): {trainable}")
    if not_trainable:
        print(f"  NOT trainable nodes ({len(not_trainable)}): {not_trainable}")
    
    # Setup data loader
    loader = Inputs(
        inputs=[X_padded],
        outputs=[y.unsqueeze(1)],
        outputs_labels=model_obj.output_dims,
        input_labels=model_obj.input_labels,
        batch_dim="s",
        batch_size=BATCH_SIZE
    )
    
    # Create NTN model
    loss = CrossEntropyLoss()
    ntn_model = NTN(
        tn=model_obj.tn,
        output_dims=model_obj.output_dims,
        input_dims=model_obj.input_dims,
        loss=loss,
        data_stream=loader
    )
    
    print(f"\nTraining {model_name}...")
    try:
        final_metrics = ntn_model.fit(
            n_epochs=N_EPOCHS,
            regularize=True,
            jitter=jitter_schedule,
            eval_metrics=CLASSIFICATION_METRICS,
            verbose=True
        )
        
        print(f"\n{'='*70}")
        print(f"RESULTS FOR {model_name}")
        print(f"{'='*70}")
        print(f"  Final Loss: {final_metrics['loss']:.4f}")
        print(f"  Final Accuracy: {final_metrics['accuracy']:.2%}")
        
        import math
        if math.isnan(final_metrics['loss']):
            print(f"  ✗ FAILED: Loss is NaN")
            return False
        elif final_metrics['accuracy'] > 0.9:
            print(f"  ✓ EXCELLENT: >90% accuracy")
            return True
        elif final_metrics['accuracy'] > 0.7:
            print(f"  ✓ GOOD: >70% accuracy")
            return True
        elif final_metrics['accuracy'] > 0.5:
            print(f"  ✓ OK: Above random baseline (33%)")
            return True
        else:
            print(f"  ✗ POOR: Close to random baseline")
            return False
            
    except Exception as e:
        print(f"\n✗ ERROR: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False

# Test 1: LMPO2
lmpo2 = LMPO2(
    L=L,
    bond_dim=BOND_DIM,
    input_dim=5,
    reduced_dim=3,
    output_dim=N_CLASSES
)
lmpo2_success = test_model("LMPO2", lmpo2)

# Test 2: MMPO2
mmpo2 = MMPO2(
    L=L,
    bond_dim=BOND_DIM,
    input_dim=5,
    output_dim=N_CLASSES,
    mask_init='identity'
)
mmpo2_success = test_model("MMPO2", mmpo2)

# Summary
print(f"\n{'='*70}")
print("SUMMARY")
print(f"{'='*70}")
print(f"  LMPO2: {'✓ SUCCESS' if lmpo2_success else '✗ FAILED'}")
print(f"  MMPO2: {'✓ SUCCESS' if mmpo2_success else '✗ FAILED'}")
print(f"{'='*70}")
