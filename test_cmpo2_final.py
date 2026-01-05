# type: ignore
"""
Final comprehensive test of CMPO2_NTN with caching
"""
import torch
import quimb.tensor as qt
from model.MPS import CMPO2_NTN
from model.builder import Inputs
from model.losses import MSELoss

print("="*70)
print("CMPO2_NTN FINAL TEST - Caching Verification")
print("="*70)

BATCH_SIZE = 10
N_SAMPLES = 50
DIM_PATCHES = 5
DIM_PIXELS = 4
BOND_DIM = 2
N_OUTPUTS = 2
N_EPOCHS = 3

print(f"\nConfiguration:")
print(f"  Sites (L): 3")
print(f"  Bond dimension: {BOND_DIM}")
print(f"  Batch size: {BATCH_SIZE}")
print(f"  Total samples: {N_SAMPLES}")
print(f"  Training epochs: {N_EPOCHS}")
print(f"  Physical dims: patches={DIM_PATCHES}, pixels={DIM_PIXELS}")

# Setup MPS
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

# Generate data
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

print(f"\nModel:")
print(f"  Type: CMPO2_NTN")
print(f"  Caching: {model.cache_environments}")
print(f"  Trainable nodes: {model._get_trainable_nodes()}")

print(f"\n" + "-"*70)
print("TRAINING")
print("-"*70)

try:
    metrics = model.fit(n_epochs=N_EPOCHS, regularize=True, jitter=1e-4, verbose=True)
    
    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)
    
    cache_stats = model.get_cache_stats()
    
    print(f"\n✓ Training completed successfully!")
    print(f"\nFinal Metrics:")
    print(f"  MSE: {metrics['mse']:.5f}")
    
    print(f"\nCache Performance:")
    print(f"  Cache hits: {cache_stats['hits']}")
    print(f"  Cache misses: {cache_stats['misses']}")
    print(f"  Total calls: {cache_stats['total']}")
    print(f"  Hit rate: {cache_stats['hit_rate']:.2%}")
    
    # Calculate expected stats
    n_batches = len(loader)
    expected_misses = N_EPOCHS * n_batches
    print(f"\nExpected Stats:")
    print(f"  Batches per epoch: {n_batches}")
    print(f"  Total batches: {N_EPOCHS * n_batches}")
    print(f"  Expected misses: ~{expected_misses} (one per batch)")
    
    if cache_stats['hits'] > 0:
        print(f"\n✓✓✓ CACHE IS WORKING! ✓✓✓")
        print(f"    Environments were reused {cache_stats['hits']} times!")
    else:
        print(f"\n✗ Cache not used")
        
except Exception as e:
    print(f"\n✗ Training failed: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*70)
