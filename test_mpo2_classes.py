# type: ignore
"""
Test the three MPO2 classes: CMPO2, LMPO2, MMPO2
"""
import torch
import numpy as np
from sklearn import datasets
import torch.nn.functional as F
from model.MPO2_models import CMPO2, LMPO2, MMPO2
from model.NTN import NTN
from model.MPS_simple import SimpleCMPO2_NTN
from model.builder import Inputs
from model.losses import CrossEntropyLoss
from model.utils import CLASSIFICATION_METRICS

torch.set_default_dtype(torch.float32)

print("="*70)
print("TESTING MPO2 CLASSES ON IRIS DATASET")
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

# Pad to 5 features and reshape for different architectures
X_padded = torch.cat([X, torch.ones(X.shape[0], 1)], dim=1)  # (150, 5)

# Configuration
N_CLASSES = 3
L = 3
BOND_DIM = 8
BATCH_SIZE = 30
N_EPOCHS = 15

print(f"\nConfiguration:")
print(f"  Sites (L): {L}")
print(f"  Bond dimension: {BOND_DIM}")
print(f"  Batch size: {BATCH_SIZE}")
print(f"  Epochs: {N_EPOCHS}")

# Jitter schedule
jitter_schedule = [5.0 * (0.1 ** epoch) for epoch in range(N_EPOCHS)]

def test_model(model_name, model_class, X_input, use_simple_cmpo2=False, **kwargs):
    """Test a single MPO2 model."""
    print(f"\n{'='*70}")
    print(f"TESTING {model_name}")
    print(f"{'='*70}")
    
    model_obj = model_class(**kwargs)
    
    print(f"\nTensor Network:")
    print(f"  Total tensors: {len(model_obj.tn.tensors)}")
    print(f"  Outer indices: {model_obj.tn.outer_inds()}")
    print(f"  Trainable nodes: {[t.tags for t in model_obj.tn.tensors if 'NOT_TRAINABLE' not in t.tags]}")
    
    # Setup data loader
    loader = Inputs(
        inputs=[X_input],
        outputs=[y.unsqueeze(1)],
        outputs_labels=model_obj.output_dims,
        input_labels=model_obj.input_labels,
        batch_dim="s",
        batch_size=BATCH_SIZE
    )
    
    # Create NTN model (use SimpleCMPO2_NTN for CMPO2)
    loss = CrossEntropyLoss()
    if use_simple_cmpo2:
        ntn_model = SimpleCMPO2_NTN(
            tn=model_obj.tn,
            output_dims=model_obj.output_dims,
            input_dims=model_obj.input_dims,
            loss=loss,
            data_stream=loader,
            psi=model_obj.psi if hasattr(model_obj, 'psi') else None,
            phi=model_obj.phi if hasattr(model_obj, 'phi') else None
        )
    else:
        ntn_model = NTN(
            tn=model_obj.tn,
            output_dims=model_obj.output_dims,
            input_dims=model_obj.input_dims,
            loss=loss,
            data_stream=loader
        )
    
    print(f"\nTraining...")
    try:
        final_metrics = ntn_model.fit(
            n_epochs=N_EPOCHS,
            regularize=True,
            jitter=jitter_schedule,
            eval_metrics=CLASSIFICATION_METRICS,
            verbose=True  # Verbose to see what's happening
        )
        
        print(f"\nResults:")
        print(f"  Final Loss: {final_metrics['loss']:.4f}")
        print(f"  Final Accuracy: {final_metrics['accuracy']:.2%}")
        
        if final_metrics['accuracy'] > 0.8:
            print(f"  ✓ SUCCESS: Good accuracy!")
        elif final_metrics['accuracy'] > 0.5:
            print(f"  ✓ OK: Above random baseline")
        else:
            print(f"  ✗ WARNING: Poor accuracy")
            
    except Exception as e:
        print(f"\n✗ ERROR: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()

# Test 1: CMPO2
print("\n" + "="*70)
print("TEST 1: CMPO2 (Cross MPO2)")
print("="*70)
# For CMPO2, reshape to (samples, patches, pixels): (150, 2, 2) -> 4 features
# But we have 5 features, so use (150, 5, 1) - 5 patches, 1 pixel each
X_cmpo2 = X_padded.unsqueeze(-1)  # (150, 5, 1)
test_model(
    "CMPO2",
    CMPO2,
    X_cmpo2,
    use_simple_cmpo2=True,  # Use SimpleCMPO2_NTN wrapper
    L=L,
    bond_dim=BOND_DIM,
    phys_dim_pixels=1,
    phys_dim_patches=5,
    output_dim=N_CLASSES
)

# Test 2: LMPO2
print("\n" + "="*70)
print("TEST 2: LMPO2 (Linear MPO2)")
print("="*70)
test_model(
    "LMPO2",
    LMPO2,
    X_padded,
    L=L,
    bond_dim=BOND_DIM,
    input_dim=5,
    reduced_dim=3,
    output_dim=N_CLASSES
)

# Test 3: MMPO2
print("\n" + "="*70)
print("TEST 3: MMPO2 (Masking MPO2)")
print("="*70)
test_model(
    "MMPO2",
    MMPO2,
    X_padded,
    L=L,
    bond_dim=BOND_DIM,
    input_dim=5,
    output_dim=N_CLASSES,
    mask_init='identity'
)

print("\n" + "="*70)
print("ALL TESTS COMPLETE")
print("="*70)
