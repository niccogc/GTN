# type: ignore
"""
Check if nodes are actually being updated
"""
import torch
import quimb.tensor as qt
from model.MPS import CMPO2_NTN
from model.builder import Inputs
from model.losses import MSELoss
from model.NTN import REGRESSION_METRICS

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

print("="*70)
print("NODE UPDATE VERIFICATION")
print("="*70)

# Get all trainable nodes
nodes = model._get_trainable_nodes()
print(f"\nTrainable nodes: {nodes}")

# Store initial state
initial_norms = {}
for node in nodes:
    tensor = model.tn[node]
    initial_norms[node] = torch.norm(tensor.data).item()
    print(f"  {node}: norm = {initial_norms[node]:.6f}")

# Do one full sweep (update all nodes once)
print(f"\nPerforming one sweep (updating all {len(nodes)} nodes)...")
columns = model._get_column_structure()
print(f"  Column structure: {columns}")

for col_idx, col_nodes in enumerate(columns):
    for node in col_nodes:
        model.update_tn_node(node, regularize=True, jitter=1e-4)

# Check if nodes changed
print(f"\nAfter one sweep:")
all_changed = True
for node in nodes:
    tensor = model.tn[node]
    new_norm = torch.norm(tensor.data).item()
    diff = abs(new_norm - initial_norms[node])
    changed = diff > 1e-10
    status = "✓ CHANGED" if changed else "✗ NO CHANGE"
    print(f"  {node}: norm = {new_norm:.6f}, diff = {diff:.6e} {status}")
    if not changed:
        all_changed = False

if all_changed:
    print(f"\n✓✓✓ ALL NODES UPDATED ✓✓✓")
else:
    print(f"\n✗✗✗ SOME NODES NOT UPDATED ✗✗✗")

# Check loss change
print(f"\nLoss verification:")
initial_scores = model.evaluate(REGRESSION_METRICS)
print(f"  MSE after sweep: {initial_scores['mse']:.6f}")

print("\n" + "="*70)
