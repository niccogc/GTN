"""
Test (MPO)² with strong regularization on polynomial task.
"""
import torch
import matplotlib.pyplot as plt
from model.MPS import MPO2_NTN, create_mpo2_tensors
from model.builder import Inputs
from model.losses import MSELoss

torch.set_default_dtype(torch.float64)

print("="*80)
print("TESTING (MPO)² WITH STRONG REGULARIZATION")
print("="*80)

# =============================================================================
# Generate Polynomial Data: y = x²
# =============================================================================

N_SAMPLES = 500
BATCH_SIZE = 100

x_raw = 2 * torch.rand(N_SAMPLES, 1) - 1  # x in [-1, 1]
y_raw = x_raw**2 + 0.05 * torch.randn(N_SAMPLES, 1)  # y = x² + noise
x_features = torch.cat([x_raw, torch.ones_like(x_raw)], dim=1)  # [x, 1]

print(f"\nDataset: y = x² + noise")
print(f"  X shape: {x_features.shape} (N_samples, features)")
print(f"  Y shape: {y_raw.shape}")
print(f"  X range: [{x_raw.min():.3f}, {x_raw.max():.3f}]")
print(f"  Y range: [{y_raw.min():.3f}, {y_raw.max():.3f}]")

# =============================================================================
# Create (MPO)² Structure
# =============================================================================

print("\n" + "="*80)
print("CREATING (MPO)² STRUCTURE")
print("="*80)

n_sites = 3
mps_bond_dim = 5
mpo_bond_dim = 5
hidden_dim = 4

print(f"\nArchitecture:")
print(f"  Sites: {n_sites}")
print(f"  MPS bond dimension: {mps_bond_dim}")
print(f"  MPO bond dimension: {mpo_bond_dim}")
print(f"  Hidden dimension: {hidden_dim} (connects MPS to MPO)")

mps_tensors, mpo_tensors = create_mpo2_tensors(
    n_sites=n_sites,
    mps_bond_dim=mps_bond_dim,
    mpo_bond_dim=mpo_bond_dim,
    lower_phys_dim=2,  # [x, 1]
    upper_phys_dim=hidden_dim,
    output_dim=1
)

print(f"\nMPS Layer (feature extraction):")
for i, t in enumerate(mps_tensors):
    print(f"  {list(t.tags)[0]}: inds={t.inds}, shape={t.shape}")

print(f"\nMPO Layer (prediction):")
for i, t in enumerate(mpo_tensors):
    print(f"  {list(t.tags)[0]}: inds={t.inds}, shape={t.shape}")

# =============================================================================
# Setup Data Loader
# =============================================================================

input_labels = [f'x{i+1}' for i in range(n_sites)]

loader = Inputs(
    inputs=[x_features],
    outputs=[y_raw],
    outputs_labels=["y"],
    input_labels=input_labels,
    batch_dim="batch",
    batch_size=BATCH_SIZE
)

# =============================================================================
# Test Different Regularization Strengths
# =============================================================================

print("\n" + "="*80)
print("TESTING DIFFERENT REGULARIZATION STRENGTHS")
print("="*80)

regularizations = [1e-2, 5e-2, 1e-1]
results = {}

for reg in regularizations:
    print(f"\n{'='*40}")
    print(f"Testing λ = {reg}")
    print('='*40)
    
    # Create fresh (MPO)² model
    mps_t, mpo_t = create_mpo2_tensors(
        n_sites=n_sites,
        mps_bond_dim=mps_bond_dim,
        mpo_bond_dim=mpo_bond_dim,
        lower_phys_dim=2,
        upper_phys_dim=hidden_dim,
        output_dim=1
    )
    
    model = MPO2_NTN.from_tensors(
        mps_tensors=mps_t,
        mpo_tensors=mpo_t,
        output_dims=["y"],
        input_dims=input_labels,
        loss=MSELoss(),
        data_stream=loader,
        method='cholesky'
    )
    
    try:
        print(f"\nTraining with λ={reg}...")
        scores = model.fit(
            n_epochs=10,
            regularize=True,
            jitter=reg,
            verbose=True
        )
        
        results[reg] = {
            'mse': scores['mse'],
            'r2': float(scores['r2_stats'][0]),
            'model': model,
            'success': True
        }
        
        print(f"\n✓ SUCCESS: MSE={scores['mse']:.6f}, R²={float(scores['r2_stats'][0]):.6f}")
        
    except Exception as e:
        print(f"\n✗ FAILED: {str(e)[:100]}...")
        results[reg] = {'success': False, 'error': str(e)}

# =============================================================================
# Summary and Best Model Evaluation
# =============================================================================

print("\n" + "="*80)
print("RESULTS SUMMARY")
print("="*80)

print(f"\n{'Regularization':<20} {'Status':<15} {'MSE':<15} {'R² Score':<15}")
print("-"*65)

best_reg = None
best_mse = float('inf')

for reg in regularizations:
    result = results[reg]
    if result['success']:
        status = "✓ Success"
        mse_str = f"{result['mse']:.6f}"
        r2_str = f"{result['r2']:.6f}"
        
        if result['mse'] < best_mse:
            best_mse = result['mse']
            best_reg = reg
    else:
        status = "✗ Failed"
        mse_str = "N/A"
        r2_str = "N/A"
    
    print(f"{reg:<20} {status:<15} {mse_str:<15} {r2_str:<15}")

if best_reg is not None:
    print(f"\n{'='*80}")
    print(f"BEST MODEL: λ = {best_reg}")
    print(f"{'='*80}")
    
    best_model = results[best_reg]['model']
    
    # Evaluate on test points
    x_test = torch.linspace(-1, 1, 100).unsqueeze(1)
    x_test_features = torch.cat([x_test, torch.ones_like(x_test)], dim=1)
    y_test_true = x_test**2
    
    # Create input tensors for prediction
    input_tensors = [
        torch.tensor(x_test_features, inds=["batch", label], tags=f"Input_{label}")
        for label in input_labels
    ]
    
    # Predict
    y_pred = best_model._batch_forward(
        input_tensors,
        best_model.tn,
        output_inds=["batch", "y"]
    )
    
    y_pred_data = y_pred.data
    
    # Compute test MSE
    test_mse = torch.mean((y_pred_data - y_test_true)**2).item()
    print(f"\nTest set MSE: {test_mse:.6f}")
    
    # Create visualization
    plt.figure(figsize=(12, 5))
    
    # Plot 1: Training data and prediction
    plt.subplot(1, 2, 1)
    plt.scatter(x_raw.numpy(), y_raw.numpy(), alpha=0.3, s=10, label='Training data')
    plt.plot(x_test.numpy(), y_test_true.numpy(), 'g-', linewidth=2, label='True: $y=x^2$')
    plt.plot(x_test.numpy(), y_pred_data.numpy(), 'r-', linewidth=2, label='(MPO)² prediction')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(f'(MPO)² Learning $y=x^2$ (λ={best_reg})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Residuals
    plt.subplot(1, 2, 2)
    residuals = (y_pred_data - y_test_true).numpy()
    plt.plot(x_test.numpy(), residuals, 'b-', linewidth=2)
    plt.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    plt.xlabel('x')
    plt.ylabel('Residual')
    plt.title('Prediction Error')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/home/nicco/Desktop/remote/GTN/mpo2_polynomial_fit.png', dpi=150)
    print(f"\n✓ Plot saved to: mpo2_polynomial_fit.png")
    plt.close()

print("\n" + "="*80)
print("TEST COMPLETE!")
print("="*80)
