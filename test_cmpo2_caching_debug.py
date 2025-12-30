# type: ignore
"""
Debug the CMPO2 caching to see what's happening with the environment.
"""
import torch
import quimb.tensor as qt
from model.MPS import CMPO2_NTN
from model.builder import Inputs
from model.losses import MSELoss

print("="*80)
print("DEBUG CMPO2 CACHING")
print("="*80)

# Simple parameters
BATCH_SIZE = 10
N_SAMPLES = 50
DIM_PATCHES = 5
DIM_PIXELS = 4
BOND_DIM = 2
N_OUTPUTS = 2

# Create MPS objects
L = 3
psi = qt.MPS_rand_state(L, bond_dim=BOND_DIM, phys_dim=DIM_PIXELS)
phi = qt.MPS_rand_state(L, bond_dim=BOND_DIM, phys_dim=DIM_PATCHES)

# Add output
middle_psi = psi['I1']
middle_psi.new_ind('out', size=N_OUTPUTS, axis=-1, mode='random', rand_strength=0.1)

psi.apply_to_arrays(lambda x: torch.tensor(x, dtype=torch.float32))
phi.apply_to_arrays(lambda x: torch.tensor(x, dtype=torch.float32))

psi.reindex({f"k{i}": f"{i}_pixels" for i in range(L)}, inplace=True)
phi.reindex({f"k{i}": f"{i}_patches" for i in range(L)}, inplace=True)

for i in range(L):
    psi.add_tag(f"{i}_Pi", where=f"I{i}")
    phi.add_tag(f"{i}_Pa", where=f"I{i}")

print("\nMPS structure:")
print("PSI (pixels):")
for i in range(L):
    t = psi[f"I{i}"]
    print(f"  {i}_Pi: {t.inds}, shape={t.shape}")

print("\nPHI (patches):")
for i in range(L):
    t = phi[f"I{i}"]
    print(f"  {i}_Pa: {t.inds}, shape={t.shape}")

tn = psi & phi

# Create data
x_data = torch.randn(N_SAMPLES, DIM_PATCHES, DIM_PIXELS)
y_data = torch.randn(N_SAMPLES, N_OUTPUTS)

input_labels_ntn = [
    [0, ("0_patches", "0_pixels")],
    [0, ("1_patches", "1_pixels")],
    [0, ("2_patches", "2_pixels")]
]
input_dims_simple = ["0", "1", "2"]

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
    input_dims=input_dims_simple,
    loss=loss,
    data_stream=loader,
    cache_environments=True
)

trainable_nodes = model._get_trainable_nodes()
print(f"\nTrainable nodes: {trainable_nodes}")

target_tag = trainable_nodes[0]  # e.g., '0_Pi'
print(f"\nWill test environment for: {target_tag}")

print("\n" + "="*80)
print("DEBUGGING THE CACHING FUNCTION")
print("="*80)

# Let's manually trace through the caching logic
from model.batch_moving_environment import BatchMovingEnvironment

# Manually create input tensors (like the loader would do)
import torch
inputs = []
for i in range(3):  # 3 sites
    # Create a tensor with indices: (batch_dim='s', '{i}_patches', '{i}_pixels')
    # Shape should be (BATCH_SIZE, patch_dim, pixel_dim)
    data = torch.randn(BATCH_SIZE, 5, 4)  # 5 for patches, 4 for pixels
    t = qt.Tensor(
        data=data,
        inds=(model.batch_dim, f'{i}_patches', f'{i}_pixels'),
        tags={f'INPUT_{i}', f'I{i}'}  # IMPORTANT: Add I{i} tag!
    )
    inputs.append(t)

# Build full_tn
full_tn = model.tn.copy()
for t in inputs:
    full_tn.add_tensor(t)

print(f"\nFull TN has {full_tn.num_tensors} tensors")

# Create environment
env_obj = BatchMovingEnvironment(
    full_tn,
    begin='left',
    bsz=1,
    batch_inds=[model.batch_dim],
    output_dims=set(model.output_dims)
)

# Get tag to position map
tag_to_pos = model._get_tag_to_position_map()
site_idx = tag_to_pos[target_tag]
print(f"\nTarget {target_tag} is at position {site_idx}")

# Move to site
env_obj.move_to(site_idx)

# Get base environment
base_env = env_obj()
print(f"\nBase environment from env_obj():")
print(f"  Num tensors: {base_env.num_tensors}")

# List tensors
print(f"  Tensors:")
for t in base_env.tensors:
    tags_str = ', '.join([tag for tag in t.tags if not tag.startswith('I')])
    print(f"    [{tags_str}]: {t.inds}, shape={t.shape}")

# Get site tensors
site_tags = env_obj.site_tag(site_idx)
print(f"\nSite tag: {site_tags}")
full_tn_at_site = env_obj.tn.select(site_tags)
print(f"Full TN at site has {full_tn_at_site.num_tensors} tensors:")
for t in full_tn_at_site.tensors:
    all_tags_str = ', '.join(sorted(t.tags))
    print(f"  Tags: [{all_tags_str}]: {t.inds}, shape={t.shape}")

# Create hole by deleting target (NEW FIX!)
final_env = base_env.copy()
final_env.delete(target_tag)
print(f"\nFinal environment after deleting {target_tag}:")
print(f"  Num tensors: {final_env.num_tensors}")

# Get outer indices
outer_inds = final_env.outer_inds()
print(f"  Outer indices: {outer_inds}")

# Check what indices we should keep
all_inds = set().union(*(t.inds for t in final_env))
print(f"  All indices in final_env: {sorted(all_inds)}")

inds_to_keep = list(outer_inds)
if model.batch_dim in all_inds and model.batch_dim not in inds_to_keep:
    inds_to_keep.append(model.batch_dim)
for out_dim in model.output_dims:
    if out_dim in all_inds and out_dim not in inds_to_keep:
        inds_to_keep.append(out_dim)

print(f"  Indices to keep: {sorted(inds_to_keep)}")

# Check bonds to target
target_tensor = model.tn[target_tag]
target_inds_set = set(target_tensor.inds)
bonds_to_target = set(outer_inds) & target_inds_set
print(f"\nBonds to target {target_tag}:")
print(f"  Outer inds: {set(outer_inds)}")
print(f"  Target inds: {target_inds_set}")
print(f"  Overlap (bonds): {bonds_to_target}")

if len(bonds_to_target) == 0:
    print(f"\n✗✗✗ PROBLEM: outer_inds doesn't include any bonds to target!")
    print(f"\nThis means the final_env was contracted incorrectly or")
    print(f"the target isn't properly connected to the environment.")
else:
    print(f"\n✓ Good! Found {len(bonds_to_target)} bond(s) to target")
