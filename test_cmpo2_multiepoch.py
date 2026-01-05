# type: ignore
"""
Test CMPO2_NTN with multiple epochs to verify training convergence.
"""
import torch
import quimb.tensor as qt
from model.MPS import CMPO2_NTN
from model.builder import Inputs
from model.losses import MSELoss

print("="*60)
print("CMPO2_NTN Multi-Epoch Test")
print("="*60)

BATCH_SIZE = 10
N_SAMPLES = 50
DIM_PATCHES = 5
DIM_PIXELS = 4
BOND_DIM = 2
N_OUTPUTS = 2
N_EPOCHS = 5

print(f"\nSetup:")
print(f"  Batch size: {BATCH_SIZE}")
print(f"  Samples: {N_SAMPLES}")
print(f"  Dimensions: patches={DIM_PATCHES}, pixels={DIM_PIXELS}")
print(f"  Bond dim: {BOND_DIM}")
print(f"  Number of epochs: {N_EPOCHS}")

L = 3
psi = qt.MPS_rand_state(L, bond_dim=BOND_DIM, phys_dim=DIM_PIXELS)
phi = qt.MPS_rand_state(L, bond_dim=BOND_DIM, phys_dim=DIM_PATCHES)

middle_psi = psi['I1']
middle_psi.new_ind('out', size=N_OUTPUTS, axis=-1, mode='random', rand_strength=0.1)
psi.apply_to_arrays(lambda x: torch.tensor(x, dtype=torch.float32))
phi.apply_to_arrays(lambda x: torch.tensor(x, dtype=torch.float32))

psi.reindex({f"k{i}": f"{i}_pixels" for i in range(L)}, inplace=True)
phi.reindex({f"k{i}": f"{i}_patches" for i in range(L)}, inplace=True)

for i in range(L):
    psi.add_tag(f"{i}_Pi", where=f"I{i}")
    phi.add_tag(f"{i}_Pa", where=f"I{i}")

tn = psi & phi

# Generate synthetic data - 3D shape (samples, patches, pixels)
torch.manual_seed(42)
x_data = torch.randn(N_SAMPLES, DIM_PATCHES, DIM_PIXELS)
y_data = torch.randn(N_SAMPLES, N_OUTPUTS)

input_labels_ntn = [
    [0, (f"{i}_patches", f"{i}_pixels")]
    for i in range(L)
]

loader = Inputs(
    inputs=[x_data],
    outputs=[y_data],
    outputs_labels=["out"],
    input_labels=input_labels_ntn,
    batch_dim="s",
    batch_size=BATCH_SIZE
)

loss = MSELoss()
model = CMPO2_NTN(
    tn=tn,
    output_dims=["out"],
    input_dims=[str(i) for i in range(L)],
    loss=loss,
    data_stream=loader,
    cache_environments=True
)

print("✓ Model created\n")

print(f"Training for {N_EPOCHS} epochs...")
print("-" * 60)

try:
    metrics = model.fit(n_epochs=N_EPOCHS, regularize=True, jitter=1e-4, verbose=True)
    
    print("\n" + "="*60)
    print("Training Results")
    print("="*60)
    print(f"✓ Training completed successfully!")
    print(f"\nFinal metrics:")
    print(f"  MSE: {metrics['mse']:.5f}")
    print(f"  R2: {metrics['r2_stats'][0]:.5f}")
    
    cache_stats = model.get_cache_stats()
    print(f"\nCache Statistics:")
    print(f"  Hits: {cache_stats['hits']}")
    print(f"  Misses: {cache_stats['misses']}")
    print(f"  Total: {cache_stats['total']}")
    print(f"  Hit rate: {cache_stats['hit_rate']:.2%}")
    
except Exception as e:
    print(f"✗ Training failed: {e}")
    import traceback
    traceback.print_exc()
