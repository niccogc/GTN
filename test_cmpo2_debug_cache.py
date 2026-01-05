# type: ignore
"""
Debug why caching isn't working
"""
import torch
import quimb.tensor as qt
from model.MPS import CMPO2_NTN
from model.builder import Inputs
from model.losses import MSELoss

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

print("="*60)
print("CACHE DEBUG")
print("="*60)

print("\n1. Model Configuration:")
print(f"   cache_environments: {model.cache_environments}")
print(f"   Model type: {type(model).__name__}")
print(f"   Has _batch_environment: {hasattr(model, '_batch_environment')}")

# Get a batch
batch_data = next(iter(loader))
print(f"\n2. Batch Structure:")
print(f"   Type: {type(batch_data)}")
if isinstance(batch_data, tuple):
    print(f"   Length: {len(batch_data)}")
    for i, item in enumerate(batch_data):
        print(f"   Item {i}: {type(item)}")
        if hasattr(item, '__len__'):
            print(f"            Length: {len(item)}")

# Get trainable nodes
nodes = model._get_trainable_nodes()
print(f"\n3. Trainable Nodes:")
print(f"   Count: {len(nodes)}")
print(f"   Nodes: {nodes}")

# Extract inputs and outputs properly
if isinstance(batch_data, tuple):
    inputs = batch_data[0]  # Usually (inputs, outputs)
    outputs = batch_data[1]
    print(f"\n4. Extracted Data:")
    print(f"   Inputs type: {type(inputs)}")
    print(f"   Outputs type: {type(outputs)}")
    if isinstance(inputs, list):
        print(f"   Inputs length: {len(inputs)}")
        for i, inp in enumerate(inputs):
            print(f"   Input {i}: {inp.inds}, shape={inp.shape}")

# Try to call _batch_environment directly
target = nodes[0]
print(f"\n5. Testing _batch_environment:")
print(f"   Target: {target}")
print(f"   Calling with inputs...")

try:
    env = model._batch_environment(inputs, model.tn, target)
    print(f"   ✓ Success!")
    print(f"   Environment shape: {env.shape}")
    print(f"   Environment inds: {env.inds}")
    
    print(f"\n6. Cache Stats After 1 Call:")
    stats = model.get_cache_stats()
    print(f"   Hits: {stats['hits']}")
    print(f"   Misses: {stats['misses']}")
    print(f"   Total: {stats['total']}")
    print(f"   Hit rate: {stats['hit_rate']:.2%}")
    
    # Call again with same inputs - should be a cache hit
    print(f"\n7. Calling Again (should hit cache)...")
    env2 = model._batch_environment(inputs, model.tn, target)
    
    stats2 = model.get_cache_stats()
    print(f"   Hits: {stats2['hits']}")
    print(f"   Misses: {stats2['misses']}")
    print(f"   Total: {stats2['total']}")
    print(f"   Hit rate: {stats2['hit_rate']:.2%}")
    
    if stats2['hits'] > stats['hits']:
        print(f"   ✓ Cache HIT detected!")
    else:
        print(f"   ✗ Cache NOT hit - investigating...")
        
except Exception as e:
    print(f"   ✗ Error: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*60)
