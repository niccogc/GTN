# type: ignore
"""
Test MPO2, LMPO2, and MMPO2 on California Housing regression
with multiple random seeds for statistical analysis.
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
print("MPO2, LMPO2, MMPO2 REGRESSION - MULTI-SEED ANALYSIS")
print("="*70)

# =============================================================================
# Configuration
# =============================================================================

N_SEEDS = 5
N_SAMPLES = 500
L = 3
BOND_DIM = 6
INPUT_DIM = 9  # 8 features + 1 bias
OUTPUT_DIM = 1
BATCH_SIZE = 100
N_EPOCHS = 10

# Jitter schedule: 0.05 with slow decay
jitter_start = 0.05
jitter_decay = 0.9
jitter_min = 1e-6
jitter_schedule = [max(jitter_start * (jitter_decay ** epoch), jitter_min) for epoch in range(N_EPOCHS)]

print(f"\nConfiguration:")
print(f"  Random seeds: {N_SEEDS}")
print(f"  Sites (L): {L}")
print(f"  Bond dimension: {BOND_DIM}")
print(f"  Input dimension: {INPUT_DIM}")
print(f"  Samples: {N_SAMPLES}")
print(f"  Batch size: {BATCH_SIZE}")
print(f"  Epochs: {N_EPOCHS}")
print(f"  Jitter: {jitter_start} * {jitter_decay}^epoch")

# =============================================================================
# Load and prepare dataset
# =============================================================================

print(f"\n1. Loading California Housing dataset...")
housing = datasets.fetch_california_housing()
X = housing.data
y = housing.target

# Storage for results
results = {
    'MPO2': {'loss': [], 'r2': [], 'seeds': []},
    'LMPO2': {'loss': [], 'r2': [], 'seeds': []},
    'MMPO2': {'loss': [], 'r2': [], 'seeds': []}
}

# =============================================================================
# Run experiments with different seeds
# =============================================================================

for seed_idx in range(N_SEEDS):
    print(f"\n" + "="*70)
    print(f"SEED {seed_idx + 1}/{N_SEEDS}")
    print("="*70)
    
    # Set seed for reproducibility
    np.random.seed(seed_idx)
    torch.manual_seed(seed_idx)
    
    # Sample subset
    indices = np.random.choice(len(X), N_SAMPLES, replace=False)
    X_subset = X[indices]
    y_subset = y[indices]
    
    # Standardize
    scaler = StandardScaler()
    X_subset = scaler.fit_transform(X_subset)
    
    # Convert to torch
    X_tensor = torch.tensor(X_subset, dtype=torch.float32)
    y_tensor = torch.tensor(y_subset, dtype=torch.float32).unsqueeze(1)
    
    print(f"  Data: {X_tensor.shape}, seed={seed_idx}")
    
    # =========================================================================
    # Test MPO2
    # =========================================================================
    
    print(f"\n  Testing MPO2...")
    try:
        mpo2 = MPO2(
            L=L,
            bond_dim=BOND_DIM,
            phys_dim=INPUT_DIM,
            output_dim=OUTPUT_DIM,
            output_site=1,
            init_strength=0.1
        )
        
        loader_mpo2 = create_inputs(
            X=X_tensor, y=y_tensor,
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
        
        scores_mpo2 = model_mpo2.fit(
            n_epochs=N_EPOCHS,
            regularize=True,
            jitter=jitter_schedule,
            eval_metrics=REGRESSION_METRICS,
            verbose=False
        )
        
        r2_mpo2 = compute_final_r2(scores_mpo2)
        loss_mpo2 = scores_mpo2['loss']
        
        # Only save if R² is positive
        if r2_mpo2 > 0:
            results['MPO2']['loss'].append(loss_mpo2)
            results['MPO2']['r2'].append(r2_mpo2)
            results['MPO2']['seeds'].append(seed_idx)
            print(f"    MPO2:  Loss={loss_mpo2:.6f}, R²={r2_mpo2:.6f} ✓")
        else:
            print(f"    MPO2:  Loss={loss_mpo2:.6f}, R²={r2_mpo2:.6f} ✗ (negative R², excluded)")
            
    except Exception as e:
        print(f"    MPO2:  FAILED - {type(e).__name__}")
    
    # =========================================================================
    # Test LMPO2
    # =========================================================================
    
    print(f"  Testing LMPO2...")
    try:
        lmpo2 = LMPO2(
            L=L,
            bond_dim=BOND_DIM,
            input_dim=INPUT_DIM,
            reduction_factor=0.5,
            output_dim=OUTPUT_DIM,
            output_site=1,
            init_strength=0.1
        )
        
        loader_lmpo2 = create_inputs(
            X=X_tensor, y=y_tensor,
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
        
        scores_lmpo2 = model_lmpo2.fit(
            n_epochs=N_EPOCHS,
            regularize=True,
            jitter=jitter_schedule,
            eval_metrics=REGRESSION_METRICS,
            verbose=False
        )
        
        r2_lmpo2 = compute_final_r2(scores_lmpo2)
        loss_lmpo2 = scores_lmpo2['loss']
        
        # Only save if R² is positive
        if r2_lmpo2 > 0:
            results['LMPO2']['loss'].append(loss_lmpo2)
            results['LMPO2']['r2'].append(r2_lmpo2)
            results['LMPO2']['seeds'].append(seed_idx)
            print(f"    LMPO2: Loss={loss_lmpo2:.6f}, R²={r2_lmpo2:.6f} ✓")
        else:
            print(f"    LMPO2: Loss={loss_lmpo2:.6f}, R²={r2_lmpo2:.6f} ✗ (negative R², excluded)")
            
    except Exception as e:
        print(f"    LMPO2: FAILED - {type(e).__name__}")
    
    # =========================================================================
    # Test MMPO2
    # =========================================================================
    
    print(f"  Testing MMPO2...")
    try:
        mmpo2 = MMPO2(
            L=L,
            bond_dim=BOND_DIM,
            input_dim=INPUT_DIM,
            output_dim=OUTPUT_DIM,
            output_site=1,
            init_strength=0.1
        )
        
        loader_mmpo2 = create_inputs(
            X=X_tensor, y=y_tensor,
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
        
        scores_mmpo2 = model_mmpo2.fit(
            n_epochs=N_EPOCHS,
            regularize=True,
            jitter=jitter_schedule,
            eval_metrics=REGRESSION_METRICS,
            verbose=False
        )
        
        r2_mmpo2 = compute_final_r2(scores_mmpo2)
        loss_mmpo2 = scores_mmpo2['loss']
        
        # Only save if R² is positive
        if r2_mmpo2 > 0:
            results['MMPO2']['loss'].append(loss_mmpo2)
            results['MMPO2']['r2'].append(r2_mmpo2)
            results['MMPO2']['seeds'].append(seed_idx)
            print(f"    MMPO2: Loss={loss_mmpo2:.6f}, R²={r2_mmpo2:.6f} ✓")
        else:
            print(f"    MMPO2: Loss={loss_mmpo2:.6f}, R²={r2_mmpo2:.6f} ✗ (negative R², excluded)")
            
    except Exception as e:
        print(f"    MMPO2: FAILED - {type(e).__name__}")

# =============================================================================
# Compute statistics
# =============================================================================

print(f"\n" + "="*70)
print("FINAL STATISTICS (excluding negative R² results)")
print("="*70)

print(f"\n{'Model':<15} {'N':<5} {'Loss (mean±std)':<25} {'R² (mean±std)':<25}")
print("-"*70)

for model_name in ['MPO2', 'LMPO2', 'MMPO2']:
    losses = results[model_name]['loss']
    r2s = results[model_name]['r2']
    n = len(losses)
    
    if n > 0:
        loss_mean = np.mean(losses)
        loss_std = np.std(losses)
        r2_mean = np.mean(r2s)
        r2_std = np.std(r2s)
        
        print(f"{model_name:<15} {n:<5} {loss_mean:.4f}±{loss_std:.4f}          {r2_mean:.4f}±{r2_std:.4f}")
    else:
        print(f"{model_name:<15} {n:<5} {'N/A':<25} {'N/A':<25}")

print("\n" + "="*70)

# Determine best model
best_model = None
best_r2_mean = -float('inf')

for model_name in ['MPO2', 'LMPO2', 'MMPO2']:
    if len(results[model_name]['r2']) > 0:
        r2_mean = np.mean(results[model_name]['r2'])
        if r2_mean > best_r2_mean:
            best_r2_mean = r2_mean
            best_model = model_name

if best_model:
    print(f"🏆 Best Model: {best_model} (R²={best_r2_mean:.4f})")
else:
    print("✗ No successful runs")

print("="*70)
