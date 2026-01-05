# type: ignore
"""
Clean CMPO2_NTN test with stable training.
"""
import torch
import quimb.tensor as qt
from model.MPS import CMPO2_NTN
from model.builder import Inputs
from model.losses import MSELoss

print("="*60)
print("CMPO2_NTN Training Test (Stable)")
print("="*60)

BATCH_SIZE = 10
N_SAMPLES = 100
DIM_PATCHES = 5
DIM_PIXELS = 4
BOND_DIM = 3
N_OUTPUTS = 2
N_EPOCHS = 10

print(f"\nSetup:")
print(f"  Batch size: {BATCH_SIZE}")
print(f"  Samples: {N_SAMPLES}")
print(f"  Epochs: {N_EPOCHS}")

L = 3
psi = qt.MPS_rand_state(L, bond_dim=BOND_DIM, phys_dim=DIM_PIXELS)
phi = qt.MPS_rand_state(L, bond_dim=BOND_DIM, phys_dim=DIM_PATCHES)

middle_psi = psi['I1']
middle_psi.new_ind('out', size=N_OUTPUTS, axis=-1, mode='random', rand_strength=0.01)
psi.apply_to_arrays(lambda x: torch.tensor(x, dtype=torch.float32))
phi.apply_to_arrays(lambda x: torch.tensor(x, dtype=torch.float32))

psi.reindex({f"k{i}": f"{i}_pixels" for i in range(L)}, inplace=True)
phi.reindex({f"k{i}": f"{i}_patches" for i in range(L)}, inplace=True)

for i in range(L):
    psi.add_tag(f"{i}_Pi", where=f"I{i}")
    phi.add_tag(f"{i}_Pa", where=f"I{i}")

tn = psi & phi

# Generate synthetic data
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

print(f"\nTraining for {N_EPOCHS} epochs...")
print("-" * 60)

try:
    metrics = model.fit(n_epochs=N_EPOCHS, regularize=False, jitter=1e-6, verbose=True)
    
    print("\n" + "="*60)
    print("Training Completed Successfully!")
    print("="*60)
    print(f"Final MSE: {metrics['mse']:.5f}")
    print(f"Final R2: {metrics['r2_stats'][0]:.5f}")
    
except Exception as e:
    print(f"✗ Training failed: {e}")
    import traceback
    traceback.print_exc()
