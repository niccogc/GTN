# type: ignore
"""
Verify the hole calculation works correctly.
env() returns LEFT + RIGHT + SAME_SITE
We need to delete the target from SAME_SITE to create the hole.
"""
import sys
sys.path.insert(0, 'model')

import torch
import quimb.tensor as qt
from batch_moving_environment import BatchMovingEnvironment

qt.set_tensor_linop_backend('torch')

print("="*80)
print("VERIFY HOLE CALCULATION")
print("="*80)

# Simple setup
L = 4
BOND_DIM = 3
PHYS_DIM = 2
BATCH_SIZE = 100

# Create MPS
psi = qt.MPS_rand_state(L, bond_dim=BOND_DIM, phys_dim=PHYS_DIM)
psi.apply_to_arrays(lambda x: torch.tensor(x, dtype=torch.float32))
psi.reindex({f"k{i}": f"phys_{i}" for i in range(L)}, inplace=True)

for i in range(L):
    psi.add_tag(f"MPS_{i}", where=f"I{i}")

print("\nMPS bond structure:")
for i in range(L):
    t = psi[f"I{i}"]
    bonds = [idx for idx in t.inds if idx.startswith('_')]
    print(f"  MPS_{i}: bonds={bonds}, physical=phys_{i}")

# Create batch inputs
inputs = []
for i in range(L):
    inp = qt.Tensor(
        data=torch.rand(BATCH_SIZE, PHYS_DIM, dtype=torch.float32),
        inds=['s', f'phys_{i}'],
        tags={f'I{i}', 'INPUT', f'INPUT_{i}'}
    )
    inputs.append(inp)

tn = psi.copy()
for inp in inputs:
    tn.add_tensor(inp)

# Initialize environment
env = BatchMovingEnvironment(tn, begin='left', bsz=1, batch_inds=['s'])

print("\n" + "="*80)
print("TESTING HOLE CALCULATION AT EACH SITE")
print("="*80)

for site_idx in range(L):
    print(f"\n{'─'*80}")
    print(f"SITE {site_idx}")
    print(f"{'─'*80}")
    
    env.move_to(site_idx)
    
    # Step 1: Get environment (LEFT + RIGHT + SAME_SITE)
    current_env = env()
    print(f"\n1. env() returns {current_env.num_tensors} tensors:")
    for t in current_env.tensors:
        tags_str = ', '.join([tag for tag in t.tags if not tag.startswith('I')])
        print(f"     [{tags_str}]")
    
    # Step 2: Delete the target to create hole
    env_with_hole = current_env.copy()
    env_with_hole.delete(f"MPS_{site_idx}")
    print(f"\n2. After deleting MPS_{site_idx}: {env_with_hole.num_tensors} tensors")
    
    # Step 3: Check outer indices (bonds to deleted target)
    outer_inds = set(env_with_hole.outer_inds())
    target = psi[f"I{site_idx}"]
    target_inds = set(target.inds)
    
    bond_inds = outer_inds & target_inds
    
    print(f"\n3. Outer indices analysis:")
    print(f"   Environment outer: {sorted(outer_inds)}")
    print(f"   Target indices: {sorted(target_inds)}")
    print(f"   Bonds (overlap): {sorted(bond_inds)}")
    
    # What bonds should we have?
    target_bonds = [idx for idx in target.inds if idx.startswith('_')]
    physical_idx = f'phys_{site_idx}'
    
    print(f"\n4. Expected bonds:")
    for bond in target_bonds:
        if bond in bond_inds:
            print(f"   ✓ {bond} (MPS bond)")
        else:
            print(f"   ✗ {bond} (MPS bond) - MISSING!")
    
    if physical_idx in bond_inds:
        print(f"   ✓ {physical_idx} (physical index)")
    else:
        print(f"   ✗ {physical_idx} (physical index) - MISSING!")
    
    # Total check
    expected_count = len(target_bonds) + 1  # MPS bonds + physical
    actual_count = len(bond_inds)
    
    print(f"\n5. Bond count: {actual_count} (expected {expected_count})")
    if actual_count == expected_count:
        print(f"   ✓✓✓ CORRECT!")
    else:
        print(f"   ✗✗✗ WRONG! Missing {expected_count - actual_count} bond(s)")
    
    # Check batch dimension
    all_inds = set()
    for t in env_with_hole.tensors:
        all_inds.update(t.inds)
    
    if 's' in all_inds:
        print(f"   ✓ Batch dimension 's' present")
    else:
        print(f"   ✗ Batch dimension 's' missing")

print("\n" + "="*80)
print("SUMMARY")
print("="*80)

print("""
The hole calculation pattern is:
  1. env() returns LEFT + RIGHT + SAME_SITE
  2. Delete target from SAME_SITE: env.delete(target_tag)
  3. outer_inds() should give bonds to the deleted target
  
If bonds are missing, the issue is in how LEFT/RIGHT are contracted.
The _get_inds_to_keep() method must preserve bonds to SAME_SITE.
""")
