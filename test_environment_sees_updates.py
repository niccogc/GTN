# type: ignore
"""
Check if the MovingEnvironment sees tensor updates
"""
import torch
import quimb.tensor as qt
from model.MPS import CMPO2_NTN
from model.builder import Inputs
from model.losses import MSELoss

BATCH_SIZE = 10
N_SAMPLES = 20  # Just 2 batches
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

print("="*70)
print("CHECKING IF MOVING ENVIRONMENT SEES UPDATES")
print("="*70)

# Get first batch
batch1 = next(iter(loader))
inputs1, outputs1 = batch1

print(f"\nBatch 1 - First input tensor ID: {id(inputs1[0])}")

# Store initial norm of node 0_Pi
initial_norm = torch.norm(model.tn['0_Pi'].data).item()
print(f"\nInitial norm of 0_Pi in model.tn: {initial_norm:.6f}")

# Call _batch_environment for node 0_Pi with batch 1 (creates cache)
print(f"\n1. Calling _batch_environment for node 0_Pi with batch 1...")
print(f"   Before _batch_environment:")
print(f"     model.tn has {len(model.tn.tensors)} tensors")
print(f"     model.tn outer inds: {model.tn.outer_inds()}")
print(f"     Has 's' in outer?: {'s' in model.tn.outer_inds()}")
env1 = model._batch_environment(inputs1, model.tn, '0_Pi')
print(f"   Created environment")
print(f"   Cache stats: {model.get_cache_stats()}")
print(f"   Environment inds: {env1.inds}")

# Check what's in the cached environment's TN
cache_key = id(inputs1[0])
cached_env_obj = model._env_cache[cache_key]
cached_tn = cached_env_obj.tn
cached_tensor_0Pi = cached_tn['0_Pi']
cached_norm = torch.norm(cached_tensor_0Pi.data).item()
print(f"   Cached TN has 0_Pi with norm: {cached_norm:.6f}")
print(f"   Cached 0_Pi tensor object ID: {id(cached_tensor_0Pi)}")
print(f"   Cached 0_Pi data ID: {id(cached_tensor_0Pi.data)}")

# Now UPDATE node 0_Pi in model.tn
print(f"\n2. Updating node 0_Pi...")
model.update_tn_node('0_Pi', regularize=True, jitter=1e-4)

# Check model.tn after update
updated_norm = torch.norm(model.tn['0_Pi'].data).item()
print(f"   Updated norm of 0_Pi in model.tn: {updated_norm:.6f}")
print(f"   Change: {abs(updated_norm - initial_norm):.6f}")

# Check cached environment's TN - does it see the update?
cached_tensor_after = cached_tn['0_Pi']
cached_norm_after = torch.norm(cached_tensor_after.data).item()
print(f"\n3. Checking cached environment's TN:")
print(f"   Cached TN still has 0_Pi with norm: {cached_norm_after:.6f}")
print(f"   Cached 0_Pi tensor object ID: {id(cached_tensor_after)}")
print(f"   Cached 0_Pi data ID: {id(cached_tensor_after.data)}")

# Check if they're the same objects
if id(cached_tensor_after) == id(cached_tensor_0Pi):
    print(f"   ✓ Same tensor object (as before update)")
else:
    print(f"   ✗ Different tensor object")

if abs(cached_norm_after - cached_norm) < 1e-10:
    print(f"\n   ✗✗✗ CACHED ENVIRONMENT HAS STALE TENSOR! ✗✗✗")
    print(f"       The cached TN still has the OLD 0_Pi tensor")
    print(f"       This is why we get wrong gradients!")
else:
    print(f"\n   ✓ Cached environment somehow got updated")

# Try calling _batch_environment again with same batch
print(f"\n4. Calling _batch_environment again for 0_Pa with same batch 1...")
env2 = model._batch_environment(inputs1, model.tn, '0_Pa')
print(f"   Cache stats: {model.get_cache_stats()}")

if model.get_cache_stats()['hits'] > 0:
    print(f"   ⚠ Cache HIT - reusing environment with stale 0_Pi!")
else:
    print(f"   ✓ Cache MISS - created fresh environment")

print("\n" + "="*70)
