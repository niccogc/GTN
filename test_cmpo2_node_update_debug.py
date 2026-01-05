# type: ignore
"""
Debug node update and environment calculation
"""
import torch
import quimb.tensor as qt
from model.MPS import CMPO2_NTN
from model.builder import Inputs
from model.losses import MSELoss
from model.NTN import REGRESSION_METRICS

print("="*70)
print("NODE UPDATE DEBUG")
print("="*70)

BATCH_SIZE = 10
N_SAMPLES = 50
DIM_PATCHES = 5
DIM_PIXELS = 4
BOND_DIM = 2
N_OUTPUTS = 2

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

print("\nModel Configuration:")
print(f"  Trainable nodes: {model._get_trainable_nodes()}")
print(f"  Caching enabled: {model.cache_environments}")

# Get initial loss
print("\nInitial evaluation...")
initial_scores = model.evaluate(REGRESSION_METRICS)
print(f"  Initial MSE: {initial_scores['mse']:.5f}")

# Do one node update
target_node = model._get_trainable_nodes()[0]
print(f"\nUpdating node: {target_node}")

# Get the node tensor before update
before_tensor = model.tn[target_node].copy()
print(f"\nBefore update:")
print(f"  Shape: {before_tensor.shape}")
print(f"  Inds: {before_tensor.inds}")
print(f"  Norm: {torch.norm(before_tensor.data):.5f}")

# Count how many times env is computed
print(f"\nPerforming update...")
print(f"  (Each batch will show ENV/TARGET/OUTPUT)")
print("-" * 70)
model.update_tn_node(target_node, regularize=True, jitter=1e-4)
print("-" * 70)

# Get the node tensor after update
after_tensor = model.tn[target_node].copy()
print(f"\nAfter update:")
print(f"  Shape: {after_tensor.shape}")
print(f"  Inds: {after_tensor.inds}")
print(f"  Norm: {torch.norm(after_tensor.data):.5f}")

# Check if it actually changed
diff = torch.norm(after_tensor.data - before_tensor.data)
print(f"\nUpdate magnitude: {diff:.5f}")

if diff < 1e-10:
    print("  ⚠ WARNING: Node barely changed!")
elif diff > 1e3:
    print("  ⚠ WARNING: Huge update (possible instability)!")
else:
    print("  ✓ Normal update magnitude")

# Evaluate again
print("\nEvaluating after update...")
after_scores = model.evaluate(REGRESSION_METRICS)
print(f"  After MSE: {after_scores['mse']:.5f}")
print(f"  Change: {after_scores['mse'] - initial_scores['mse']:.5f}")

if after_scores['mse'] < initial_scores['mse']:
    print("  ✓ Loss improved!")
elif after_scores['mse'] > initial_scores['mse'] * 10:
    print("  ✗ Loss exploded!")
else:
    print("  ~ Loss slightly worse (can happen with Newton)")

# Check cache stats
stats = model.get_cache_stats()
print(f"\nCache stats:")
print(f"  Hits: {stats['hits']}")
print(f"  Misses: {stats['misses']}")
print(f"  Total calls: {stats['total']}")
print(f"  Hit rate: {stats['hit_rate']:.2%}")

expected_calls = len(loader)  # One per batch
print(f"\nExpected calls: {expected_calls} (number of batches)")
print(f"Actual calls: {stats['total']}")

if stats['total'] == expected_calls:
    print("  ✓ Correct number of environment computations")
    if stats['hits'] == expected_calls - 1:
        print("  ✓ Cache working perfectly (1 miss to create, rest hits)")
    elif stats['misses'] == expected_calls:
        print("  ✗ No cache hits! Cache not being used!")
else:
    print(f"  ⚠ Unexpected number of calls")

print("\n" + "="*70)
