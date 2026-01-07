# type: ignore
"""
Test MPO2, LMPO2, and MMPO2 on real regression dataset with decaying jitter.

Uses California Housing dataset: predict median house value from 8 features.
"""
import torch
import numpy as np
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from model.NTN import NTN
from model.losses import MSELoss
from model.utils import REGRESSION_METRICS, compute_final_r2, create_inputs
from model.MPO2_models import MPO2, LMPO2, MMPO2

torch.set_default_dtype(torch.float32)

print("="*70)
print("TESTING MPO2, LMPO2, MMPO2 ON CALIFORNIA HOUSING REGRESSION")
print("="*70)

# =============================================================================
# Load California Housing Dataset
# =============================================================================

print("\n1. Loading California Housing dataset...")
housing = datasets.fetch_california_housing()
X = housing.data  # (20640, 8) - 8 features
y = housing.target  # (20640,) - median house value

print(f"  Dataset: {len(X)} samples, {X.shape[1]} features")
print(f"  Features: {housing.feature_names}")
print(f"  Target: Median house value (in $100,000s)")
print(f"  X range: min={X.min():.3f}, max={X.max():.3f}")
print(f"  y range: min={y.min():.3f}, max={y.max():.3f}")

# Use subset for faster testing
N_SAMPLES = 500
indices = np.random.choice(len(X), N_SAMPLES, replace=False)
X = X[indices]
y = y[indices]

# Standardize features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Convert to torch
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

print(f"\n  Using {N_SAMPLES} samples")
print(f"  X shape (before bias): {X.shape} (samples, features)")
print(f"  y shape: {y.shape}")

# Configuration
L = 3  # 3 sites
BOND_DIM = 6
INPUT_DIM = 9  # 8 housing features + 1 bias (appended by create_inputs)
OUTPUT_DIM = 1  # scalar output (house value)
BATCH_SIZE = 100
N_EPOCHS = 10

print(f"\nConfiguration:")
print(f"  Sites (L): {L}")
print(f"  Bond dimension: {BOND_DIM}")
print(f"  Input dimension: {INPUT_DIM}")
print(f"  Output dimension: {OUTPUT_DIM}")
print(f"  Batch size: {BATCH_SIZE}")
print(f"  Training epochs: {N_EPOCHS}")

# Jitter schedule: start with 0.05 and slow decay
jitter_start = 0.05
jitter_decay = 0.9
jitter_min = 1e-6
jitter_schedule = [max(jitter_start * (jitter_decay ** epoch), jitter_min) for epoch in range(N_EPOCHS)]

print(f"\n  Jitter schedule: {jitter_start} * {jitter_decay}^epoch (min={jitter_min})")
for i in [0, min(4, N_EPOCHS-1)]:
    print(f"    Epoch {i+1}: {jitter_schedule[i]:.6f}")

# =============================================================================
# Test 1: MPO2 (Simple MPS)
# =============================================================================

print(f"\n" + "="*70)
print("TEST 1: MPO2 (Simple MPS with output dimension)")
print("="*70)

mpo2 = MPO2(
    L=L,
    bond_dim=BOND_DIM,
    phys_dim=INPUT_DIM,
    output_dim=OUTPUT_DIM,
    output_site=2,  # Middle-ish site
    init_strength=0.1
)

print(f"\nMPO2 structure:")
print(f"  Input labels: {mpo2.input_labels}")
print(f"  Total tensors: {len(mpo2.tn.tensors)}")

loader_mpo2 = create_inputs(
    X=X,
    y=y,
    input_labels=mpo2.input_labels,
    output_labels=mpo2.output_dims,
    batch_size=BATCH_SIZE
)

model_mpo2 = NTN(
    tn=mpo2.tn,
    output_dims=mpo2.output_dims,
    input_dims=mpo2.input_dims,
    loss=MSELoss(),
    data_stream=loader_mpo2
)

print(f"\nTraining MPO2...")
try:
    scores_mpo2 = model_mpo2.fit(
        n_epochs=N_EPOCHS,
        regularize=True,
        jitter=jitter_schedule,
        eval_metrics=REGRESSION_METRICS,
        verbose=True
    )
    
    print(f"\n✓ MPO2 Training Complete:")
    print(f"  Final Loss (MSE): {scores_mpo2['loss']:.6f}")
    r2_mpo2 = compute_final_r2(scores_mpo2)
    print(f"  Final R²: {r2_mpo2:.6f}")
    mpo2_success = True
    
except Exception as e:
    print(f"\n✗ MPO2 Training FAILED:")
    print(f"  {type(e).__name__}: {e}")
    mpo2_success = False

# =============================================================================
# Test 2: LMPO2 (Linear MPO2 with 50% reduction)
# =============================================================================

print(f"\n" + "="*70)
print("TEST 2: LMPO2 (MPO reduction + MPS)")
print("="*70)

lmpo2 = LMPO2(
    L=L,
    bond_dim=BOND_DIM,
    input_dim=INPUT_DIM,
    reduction_factor=0.5,  # 50% reduction: 8 -> 4
    output_dim=OUTPUT_DIM,
    output_site=2,
    init_strength=0.1
)

print(f"\nLMPO2 structure:")
print(f"  Input labels: {lmpo2.input_labels}")
print(f"  Reduction: {lmpo2.input_dim} -> {lmpo2.reduced_dim} ({lmpo2.reduction_factor:.0%})")
print(f"  Total tensors: {len(lmpo2.tn.tensors)} (4 MPO + 4 MPS)")

loader_lmpo2 = create_inputs(
    X=X,
    y=y,
    input_labels=lmpo2.input_labels,
    output_labels=lmpo2.output_dims,
    batch_size=BATCH_SIZE
)

model_lmpo2 = NTN(
    tn=lmpo2.tn,
    output_dims=lmpo2.output_dims,
    input_dims=lmpo2.input_dims,
    loss=MSELoss(),
    data_stream=loader_lmpo2
)

print(f"\nTraining LMPO2...")
try:
    scores_lmpo2 = model_lmpo2.fit(
        n_epochs=N_EPOCHS,
        regularize=True,
        jitter=jitter_schedule,
        eval_metrics=REGRESSION_METRICS,
        verbose=True
    )
    
    print(f"\n✓ LMPO2 Training Complete:")
    print(f"  Final Loss (MSE): {scores_lmpo2['loss']:.6f}")
    r2_lmpo2 = compute_final_r2(scores_lmpo2)
    print(f"  Final R²: {r2_lmpo2:.6f}")
    lmpo2_success = True
    
except Exception as e:
    print(f"\n✗ LMPO2 Training FAILED:")
    print(f"  {type(e).__name__}: {e}")
    lmpo2_success = False

# =============================================================================
# Test 3: MMPO2 (Masking MPO + MPS)
# =============================================================================

print(f"\n" + "="*70)
print("TEST 3: MMPO2 (Cumulative mask MPO + MPS)")
print("="*70)

mmpo2 = MMPO2(
    L=L,
    bond_dim=BOND_DIM,
    input_dim=INPUT_DIM,
    output_dim=OUTPUT_DIM,
    output_site=2,
    init_strength=0.1
)

print(f"\nMMPO2 structure:")
print(f"  Input labels: {mmpo2.input_labels}")
print(f"  Total tensors: {len(mmpo2.tn.tensors)} (4 Mask MPO + 4 MPS)")
print(f"  Mask bond dim: {mmpo2.input_dim} (= input_dim)")
print(f"  MPS bond dim: {mmpo2.bond_dim}")

loader_mmpo2 = create_inputs(
    X=X,
    y=y,
    input_labels=mmpo2.input_labels,
    output_labels=mmpo2.output_dims,
    batch_size=BATCH_SIZE
)

model_mmpo2 = NTN(
    tn=mmpo2.tn,
    output_dims=mmpo2.output_dims,
    input_dims=mmpo2.input_dims,
    loss=MSELoss(),
    data_stream=loader_mmpo2
)

print(f"\nTraining MMPO2...")
try:
    scores_mmpo2 = model_mmpo2.fit(
        n_epochs=N_EPOCHS,
        regularize=True,
        jitter=jitter_schedule,
        eval_metrics=REGRESSION_METRICS,
        verbose=True
    )
    
    print(f"\n✓ MMPO2 Training Complete:")
    print(f"  Final Loss (MSE): {scores_mmpo2['loss']:.6f}")
    r2_mmpo2 = compute_final_r2(scores_mmpo2)
    print(f"  Final R²: {r2_mmpo2:.6f}")
    mmpo2_success = True
    
except Exception as e:
    print(f"\n✗ MMPO2 Training FAILED:")
    print(f"  {type(e).__name__}: {e}")
    mmpo2_success = False

# =============================================================================
# Summary
# =============================================================================

print(f"\n" + "="*70)
print("SUMMARY")
print("="*70)

print(f"\n{'Model':<15} {'Status':<15} {'Final Loss':<15} {'Final R²':<15}")
print("-"*60)

if mpo2_success:
    print(f"{'MPO2':<15} {'✓ Success':<15} {scores_mpo2['loss']:<15.6f} {r2_mpo2:<15.6f}")
else:
    print(f"{'MPO2':<15} {'✗ Failed':<15} {'N/A':<15} {'N/A':<15}")

if lmpo2_success:
    print(f"{'LMPO2':<15} {'✓ Success':<15} {scores_lmpo2['loss']:<15.6f} {r2_lmpo2:<15.6f}")
else:
    print(f"{'LMPO2':<15} {'✗ Failed':<15} {'N/A':<15} {'N/A':<15}")

if mmpo2_success:
    print(f"{'MMPO2':<15} {'✓ Success':<15} {scores_mmpo2['loss']:<15.6f} {r2_mmpo2:<15.6f}")
else:
    print(f"{'MMPO2':<15} {'✗ Failed':<15} {'N/A':<15} {'N/A':<15}")

print(f"\n" + "="*70)

# Check for any failures
all_success = mpo2_success and lmpo2_success and mmpo2_success
if all_success:
    print("✓ ALL MODELS TRAINED SUCCESSFULLY!")
    
    # Determine best model
    best_model = None
    best_r2 = -float('inf')
    
    if mpo2_success and r2_mpo2 > best_r2:
        best_model = "MPO2"
        best_r2 = r2_mpo2
        best_loss = scores_mpo2['loss']
    
    if lmpo2_success and r2_lmpo2 > best_r2:
        best_model = "LMPO2"
        best_r2 = r2_lmpo2
        best_loss = scores_lmpo2['loss']
    
    if mmpo2_success and r2_mmpo2 > best_r2:
        best_model = "MMPO2"
        best_r2 = r2_mmpo2
        best_loss = scores_mmpo2['loss']
    
    print(f"\n🏆 Best Model: {best_model} (R²={best_r2:.6f}, Loss={best_loss:.6f})")
else:
    failed_models = []
    if not mpo2_success:
        failed_models.append("MPO2")
    if not lmpo2_success:
        failed_models.append("LMPO2")
    if not mmpo2_success:
        failed_models.append("MMPO2")
    print(f"✗ FAILED MODELS: {', '.join(failed_models)}")

print("="*70)
