# type: ignore
"""
Verify that node updates actually modify self.tn
"""
import torch
import quimb.tensor as qt
from model.MPS import CMPO2_NTN
from model.builder import Inputs
from model.losses import MSELoss
from model.NTN import REGRESSION_METRICS

print("="*70)
print("VERIFY NODE UPDATES IN self.tn")
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

target_node = '0_Pi'
print(f"\nTarget node: {target_node}")

# Get initial tensor from model.tn
initial_tensor = model.tn[target_node].copy()
print(f"\nInitial state (from model.tn):")
print(f"  Shape: {initial_tensor.shape}")
print(f"  Inds: {initial_tensor.inds}")
print(f"  Data ID: {id(initial_tensor.data)}")
print(f"  Norm: {torch.norm(initial_tensor.data):.6f}")
print(f"  First few values: {initial_tensor.data.flatten()[:5]}")

# Store reference to the actual tensor object in the TN
tn_tensor_ref = model.tn[target_node]
print(f"\nTensor object in TN:")
print(f"  Object ID: {id(tn_tensor_ref)}")
print(f"  Data ID: {id(tn_tensor_ref.data)}")

# Perform update
print(f"\n" + "="*70)
print("PERFORMING UPDATE")
print("="*70)
model.update_tn_node(target_node, regularize=True, jitter=1e-4)

# Check the tensor in model.tn after update
after_tensor = model.tn[target_node]
print(f"\nAfter update (from model.tn):")
print(f"  Object ID: {id(after_tensor)}")
print(f"  Data ID: {id(after_tensor.data)}")
print(f"  Shape: {after_tensor.shape}")
print(f"  Inds: {after_tensor.inds}")
print(f"  Norm: {torch.norm(after_tensor.data):.6f}")
print(f"  First few values: {after_tensor.data.flatten()[:5]}")

# Check if it's a new object or modified in-place
if id(after_tensor) == id(tn_tensor_ref):
    print(f"\n  ⚠ SAME tensor object (might be in-place modification)")
else:
    print(f"\n  ✓ NEW tensor object (replaced in TN)")

if id(after_tensor.data) == id(initial_tensor.data):
    print(f"  ⚠ SAME data array (in-place modification of data)")
else:
    print(f"  ✓ NEW data array")

# Check if values actually changed
diff = torch.norm(after_tensor.data - initial_tensor.data)
print(f"\n  Update magnitude: {diff:.6f}")

if diff > 1e-10:
    print(f"  ✓✓✓ VALUES CHANGED ✓✓✓")
else:
    print(f"  ✗✗✗ VALUES DID NOT CHANGE ✗✗✗")

# Verify with another lookup
verify_tensor = model.tn[target_node]
print(f"\nVerification (another lookup):")
print(f"  Same as after_tensor: {id(verify_tensor) == id(after_tensor)}")
print(f"  Norm: {torch.norm(verify_tensor.data):.6f}")

print("\n" + "="*70)
