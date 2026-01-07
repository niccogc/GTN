# type: ignore
"""
Test MPO2, LMPO2, and MMPO2 on polynomial regression task: y = x²
"""
import torch
import numpy as np
from model.NTN import NTN
from model.builder import Inputs
from model.losses import MSELoss
from model.utils import REGRESSION_METRICS, compute_final_r2
from model.MPO2_models import MPO2, LMPO2, MMPO2

torch.set_default_dtype(torch.float32)

print("="*70)
print("TESTING MPO2, LMPO2, MMPO2 ON POLYNOMIAL REGRESSION")
print("="*70)

# =============================================================================
# Generate Polynomial Data: y = x²
# =============================================================================

print("\n1. Generating polynomial dataset: y = x²")
N_SAMPLES = 500
BATCH_SIZE = 100

x_raw = 2 * torch.rand(N_SAMPLES, 1) - 1  # x in [-1, 1]
y_raw = x_raw**2 + 0.05 * torch.randn(N_SAMPLES, 1)  # y = x² + noise
x_features = torch.cat([x_raw, torch.ones_like(x_raw)], dim=1)  # [x, 1] -> 2 features

print(f"  Dataset: y = x² + noise")
print(f"  X shape: {x_features.shape} (samples, features)")
print(f"  Y shape: {y_raw.shape}")
print(f"  X range: [{x_raw.min():.3f}, {x_raw.max():.3f}]")
print(f"  Y range: [{y_raw.min():.3f}, {y_raw.max():.3f}]")

# Configuration
L = 3  # 3 sites
BOND_DIM = 5
INPUT_DIM = 2  # [x, 1]
OUTPUT_DIM = 1  # scalar output
N_EPOCHS = 15

print(f"\nConfiguration:")
print(f"  Sites (L): {L}")
print(f"  Bond dimension: {BOND_DIM}")
print(f"  Input dimension: {INPUT_DIM}")
print(f"  Output dimension: {OUTPUT_DIM}")
print(f"  Batch size: {BATCH_SIZE}")
print(f"  Training epochs: {N_EPOCHS}")

# Jitter schedule: constant small jitter
jitter_schedule = 1e-4  # Constant jitter (increased from 1e-6)
print(f"  Jitter: {jitter_schedule} (constant)")

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
    output_site=1,  # Middle site
    init_strength=0.1
)

print(f"\nMPO2 structure:")
print(f"  Input labels: {mpo2.input_labels}")
print(f"  Input dims: {mpo2.input_dims}")
print(f"  Output dims: {mpo2.output_dims}")
print(f"  Total tensors: {len(mpo2.tn.tensors)}")

loader_mpo2 = Inputs(
    inputs=[x_features],
    outputs=[y_raw],
    outputs_labels=mpo2.output_dims,
    input_labels=mpo2.input_labels,
    batch_dim="s",
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
    reduction_factor=0.5,  # 50% reduction: 2 -> 1
    output_dim=OUTPUT_DIM,
    output_site=1,
    init_strength=0.1
)

print(f"\nLMPO2 structure:")
print(f"  Input labels: {lmpo2.input_labels}")
print(f"  Input dims: {lmpo2.input_dims}")
print(f"  Output dims: {lmpo2.output_dims}")
print(f"  Reduction: {lmpo2.input_dim} -> {lmpo2.reduced_dim} ({lmpo2.reduction_factor:.0%})")
print(f"  Total tensors: {len(lmpo2.tn.tensors)} (3 MPO + 3 MPS)")

loader_lmpo2 = Inputs(
    inputs=[x_features],
    outputs=[y_raw],
    outputs_labels=lmpo2.output_dims,
    input_labels=lmpo2.input_labels,
    batch_dim="s",
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
    output_site=1,
    init_strength=0.1
)

print(f"\nMMPO2 structure:")
print(f"  Input labels: {mmpo2.input_labels}")
print(f"  Input dims: {mmpo2.input_dims}")
print(f"  Output dims: {mmpo2.output_dims}")
print(f"  Total tensors: {len(mmpo2.tn.tensors)} (3 Mask MPO + 3 MPS)")
print(f"  Mask bond dim: {mmpo2.input_dim} (= input_dim)")
print(f"  MPS bond dim: {mmpo2.bond_dim}")

loader_mmpo2 = Inputs(
    inputs=[x_features],
    outputs=[y_raw],
    outputs_labels=mmpo2.output_dims,
    input_labels=mmpo2.input_labels,
    batch_dim="s",
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

print(f"\n{'Model':<15} {'Status':<15} {'Final MSE':<15} {'Final R²':<15}")
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
