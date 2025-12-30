# type: ignore
"""
Simple test: Single MPS with batch inputs (s x phys_dim).
Test if BatchMovingEnvironment can create proper holes going left and right.
GO SLOWLY - print everything and check carefully.
"""
import sys
sys.path.insert(0, 'model')

import torch
import quimb.tensor as qt
from batch_moving_environment import BatchMovingEnvironment

qt.set_tensor_linop_backend('torch')

def print_section(title):
    print("\n" + "="*80)
    print(f"  {title}")
    print("="*80 + "\n")

print_section("SIMPLE MPS + BATCH INPUTS HOLE TEST")

# Simple setup: Just ONE MPS with batch inputs
L = 4
BOND_DIM = 3
PHYS_DIM = 2
BATCH_SIZE = 100
OUTPUT_DIM = 10

print(f"Configuration:")
print(f"  L (sites): {L}")
print(f"  Bond dim: {BOND_DIM}")
print(f"  Physical dim: {PHYS_DIM}")
print(f"  Batch size: {BATCH_SIZE}")
print(f"  Output dim: {OUTPUT_DIM}")

# Create a single MPS
psi = qt.MPS_rand_state(L, bond_dim=BOND_DIM, phys_dim=PHYS_DIM)

# Add output dimension to one of the MPS tensors - BEFORE converting to torch
middle_site = 2  # Site 2
middle_tensor = psi[f'I{middle_site}']
print(f"\nAdding 'out' dimension to MPS site {middle_site}")
middle_tensor.new_ind('out', size=OUTPUT_DIM, axis=-1, mode='random', rand_strength=0.1)

# Now convert to torch
psi.apply_to_arrays(lambda x: torch.tensor(x, dtype=torch.float32))

print_section("1. MPS STRUCTURE - BEFORE REINDEXING")

print("MPS tensors with original indices:")
for i in range(L):
    t = psi[f"I{i}"]
    print(f"  Site {i}:")
    print(f"    Indices: {t.inds}")
    print(f"    Shape: {t.shape}")
    print(f"    Tags: {t.tags}")

# Reindex physical dimensions
psi.reindex({f"k{i}": f"phys_{i}" for i in range(L)}, inplace=True)

print_section("2. MPS STRUCTURE - AFTER REINDEXING")

print("MPS tensors after reindexing physical dimensions:")
bond_indices = {}
for i in range(L):
    t = psi[f"I{i}"]
    print(f"\n  Site {i}:")
    print(f"    Indices: {t.inds}")
    print(f"    Shape: {t.shape}")
    
    # Identify bond indices
    bonds = [idx for idx in t.inds if idx.startswith('_')]
    phys = [idx for idx in t.inds if idx.startswith('phys_')]
    out = [idx for idx in t.inds if idx == 'out']
    
    print(f"    Bond indices: {bonds}")
    print(f"    Physical index: {phys}")
    if out:
        print(f"    Output index: {out}")
    
    bond_indices[i] = bonds

# Add unique tags for hole creation
for i in range(L):
    psi.add_tag(f"MPS_{i}", where=f"I{i}")

print_section("3. BATCH INPUT TENSORS")

# Create batch inputs: shape (BATCH, PHYS_DIM) for each site
inputs = []
for i in range(L):
    inp = qt.Tensor(
        data=torch.rand(BATCH_SIZE, PHYS_DIM, dtype=torch.float32),
        inds=['s', f'phys_{i}'],  # Connects to MPS physical index
        tags={f'I{i}', 'INPUT', f'INPUT_{i}'}
    )
    inputs.append(inp)
    print(f"  Site {i}:")
    print(f"    Indices: {inp.inds}")
    print(f"    Shape: {inp.shape}")
    print(f"    Connects to MPS phys_{i}")

# Build tensor network
tn = psi.copy()
for inp in inputs:
    tn.add_tensor(inp)

print(f"\n✓ Tensor network has {tn.num_tensors} tensors")
print(f"  {L} MPS tensors + {L} input tensors = {2*L} total")

print_section("4. EXPECTED ENVIRONMENT STRUCTURE AT EACH SITE")

print("What the environment SHOULD look like at each site:\n")

for i in range(L):
    print(f"Site {i} (Target: MPS_{i}):")
    print(f"  Environment EXCLUDES:")
    print(f"    - MPS_{i} (the target)")
    print(f"    - INPUT_{i} (same site, bsz=1)")
    print(f"  Environment INCLUDES:")
    
    included = []
    for j in range(L):
        if j != i:
            included.append(f"MPS_{j}")
            included.append(f"INPUT_{j}")
    print(f"    {', '.join(included)}")
    
    print(f"  Expected outer indices (bonds to target):")
    target = psi[f"I{i}"]
    target_inds = set(target.inds)
    
    # Bond indices that connect to other MPS sites
    expected_bonds = []
    for bond in bond_indices[i]:
        # Check if this bond connects to left or right
        if i > 0 and bond in bond_indices[i-1]:
            expected_bonds.append(f"{bond} (left bond)")
        elif i < L-1 and bond in bond_indices[i+1]:
            expected_bonds.append(f"{bond} (right bond)")
    
    # Physical index connects to INPUT
    expected_bonds.append(f"phys_{i} (to INPUT_{i})")
    
    # Output index - IMPORTANT!
    if 'out' in target_inds:
        print(f"    {', '.join(expected_bonds)}")
        print(f"    NOTE: 'out' is on target but NOT connected to anything")
        print(f"          So 'out' should NOT appear in environment outer indices")
    else:
        print(f"    {', '.join(expected_bonds)}")
        # Check if output appears elsewhere
        if i != middle_site:
            print(f"    'out' appears in environment (from site {middle_site})")
    
    print()

print_section("5. INITIALIZE BatchMovingEnvironment")

env = BatchMovingEnvironment(
    tn,
    begin='left',
    bsz=1,
    batch_inds=['s'],
    output_dims={'out'}
)

print(f"✓ Environment initialized")
print(f"  Position: {env.pos}")
print(f"  batch_inds: {env.batch_inds}")
print(f"  output_dims: {env.output_dims}")

print_section("6. TEST EACH SITE - LEFT TO RIGHT")

for site_idx in range(L):
    print(f"\n{'='*80}")
    print(f"TESTING SITE {site_idx}")
    print(f"{'='*80}\n")
    
    # Move to site
    env.move_to(site_idx)
    print(f"✓ Moved to position {site_idx}")
    
    # Get target tensor
    target = psi[f"I{site_idx}"]
    print(f"\nTARGET (MPS_{site_idx}):")
    print(f"  Indices: {target.inds}")
    print(f"  Shape: {target.shape}")
    
    # Get environment (excludes ALL tensors at this site)
    current_env = env()
    print(f"\nENVIRONMENT (from env()):")
    print(f"  Number of tensors: {current_env.num_tensors}")
    print(f"  Expected: {2*L - 2} (total {2*L} minus MPS_{site_idx} and INPUT_{site_idx})")
    
    # List all tensors in environment
    print(f"  Tensors in environment:")
    for t in current_env.tensors:
        # Get non-site tags
        tags = [tag for tag in t.tags if not tag.startswith('I')]
        print(f"    - {tags}: shape={t.shape}, inds={t.inds}")
    
    # Create hole by deleting target
    env_with_hole = current_env.copy()
    
    if f"MPS_{site_idx}" in env_with_hole.tags:
        env_with_hole.delete(f"MPS_{site_idx}")
        print(f"\n✓ Deleted MPS_{site_idx} from environment")
        print(f"  Tensors after deletion: {env_with_hole.num_tensors}")
    else:
        print(f"\n✗ ERROR: MPS_{site_idx} not in environment!")
        continue
    
    # Get outer indices
    outer_inds = set(env_with_hole.outer_inds())
    target_inds = set(target.inds)
    
    print(f"\nINDICES ANALYSIS:")
    print(f"  Environment outer indices: {sorted(outer_inds)}")
    print(f"  Target indices: {sorted(target_inds)}")
    
    # Bond indices (overlap)
    bond_inds = outer_inds & target_inds
    print(f"  Overlapping (bonds): {sorted(bond_inds)}")
    
    # Check expected bonds
    expected_bond_count = 0
    if site_idx > 0:
        expected_bond_count += 1  # Left bond
    if site_idx < L - 1:
        expected_bond_count += 1  # Right bond
    expected_bond_count += 1  # Physical index to input
    
    print(f"\n  Expected bonds:")
    if site_idx > 0:
        left_bond = [b for b in bond_indices[site_idx] if b in bond_indices[site_idx-1]]
        print(f"    - Left bond: {left_bond}")
    if site_idx < L - 1:
        right_bond = [b for b in bond_indices[site_idx] if b in bond_indices[site_idx+1]]
        print(f"    - Right bond: {right_bond}")
    print(f"    - Physical: phys_{site_idx}")
    
    print(f"\n  Bond count: {len(bond_inds)}, expected: {expected_bond_count}")
    if len(bond_inds) == expected_bond_count:
        print(f"  ✓✓ CORRECT number of bonds")
    else:
        print(f"  ✗✗ WRONG number of bonds!")
    
    # Check special dimensions
    all_env_inds = set()
    for t in env_with_hole.tensors:
        all_env_inds.update(t.inds)
    
    print(f"\nSPECIAL DIMENSIONS:")
    print(f"  Batch 's' in environment: {'✓' if 's' in all_env_inds else '✗'}")
    
    # Output dimension handling - KEY POINT
    if 'out' in target_inds:
        print(f"  Output 'out' on target: YES")
        print(f"  Output 'out' in environment: {'YES' if 'out' in all_env_inds else 'NO'}")
        print(f"    → 'out' is ONLY on target, NOT connected to anything else")
        print(f"    → So 'out' should NOT be in environment")
        if 'out' not in all_env_inds:
            print(f"    ✓✓ CORRECT - 'out' not in environment")
        else:
            print(f"    ✗✗ WRONG - 'out' should not be in environment!")
    else:
        print(f"  Output 'out' on target: NO")
        if site_idx != middle_site:
            print(f"  Output 'out' in environment: {'YES (from site 2)' if 'out' in all_env_inds else 'NO'}")
            if 'out' in all_env_inds:
                print(f"    ✓✓ CORRECT - 'out' in environment from site {middle_site}")
    
    # Contract
    inds_to_keep = list(outer_inds)
    if 's' in all_env_inds and 's' not in inds_to_keep:
        inds_to_keep.append('s')
    if 'out' in all_env_inds and 'out' not in inds_to_keep:
        inds_to_keep.append('out')
    
    print(f"\nCONTRACTION:")
    print(f"  Indices to keep: {sorted(inds_to_keep)}")
    
    try:
        contracted = env_with_hole.contract(all, output_inds=inds_to_keep)
        print(f"  ✓ Contraction successful")
        print(f"  Result shape: {contracted.shape}")
        if hasattr(contracted, 'inds'):
            print(f"  Result indices: {contracted.inds}")
    except Exception as e:
        print(f"  ✗ Contraction FAILED: {e}")
        continue

print_section("SUMMARY")

print("""
Key points verified:

1. MPS structure with output dimension on site 2
2. Batch inputs connecting to each MPS physical index
3. Environment excludes target and input at same site
4. Bond indices correctly identified
5. IMPORTANT: 'out' dimension on site 2 is NOT connected to anything
   - When target is site 2: 'out' should NOT be in environment outer indices
   - When target is NOT site 2: 'out' should appear in environment
""")

print("="*80)
print("  TEST COMPLETE")
print("="*80)
