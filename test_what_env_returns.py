# type: ignore
"""
Check what env() actually returns - does it give us the raw environment?
Or something else?
"""
import sys
sys.path.insert(0, 'model')

import torch
import quimb.tensor as qt
from batch_moving_environment import BatchMovingEnvironment

qt.set_tensor_linop_backend('torch')

# Simple setup
L = 4
BOND_DIM = 3
PHYS_DIM = 2
BATCH_SIZE = 100

print("="*80)
print("WHAT DOES env() RETURN?")
print("="*80)

# Create MPS
psi = qt.MPS_rand_state(L, bond_dim=BOND_DIM, phys_dim=PHYS_DIM)
psi.apply_to_arrays(lambda x: torch.tensor(x, dtype=torch.float32))
psi.reindex({f"k{i}": f"phys_{i}" for i in range(L)}, inplace=True)

for i in range(L):
    psi.add_tag(f"MPS_{i}", where=f"I{i}")

print("\nOriginal MPS:")
for i in range(L):
    t = psi[f"I{i}"]
    print(f"  MPS_{i}: indices={t.inds}")

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

print(f"\nTotal TN has {tn.num_tensors} tensors")

# Initialize environment
env = BatchMovingEnvironment(tn, begin='left', bsz=1, batch_inds=['s'])

print("\n" + "="*80)
print("TESTING EACH SITE")
print("="*80)

for site_idx in range(L):
    print(f"\n{'─'*80}")
    print(f"SITE {site_idx}")
    print(f"{'─'*80}")
    
    env.move_to(site_idx)
    
    print(f"\nWhat env() returns at position {site_idx}:")
    current_env = env()
    
    print(f"  Type: {type(current_env)}")
    print(f"  Number of tensors: {current_env.num_tensors}")
    
    print(f"\n  Tensors in env():")
    for t in current_env.tensors:
        tags_str = ', '.join([tag for tag in t.tags if not tag.startswith('I')])
        print(f"    [{tags_str}]: shape={t.shape}, inds={t.inds}")
    
    print(f"\n  What SHOULD be in environment at site {site_idx}:")
    print(f"    According to documentation:")
    print(f"      - LEFT: All sites < {site_idx} (contracted)")
    print(f"      - RIGHT: All sites > {site_idx} (contracted)")
    print(f"      - SAME SITE: Excluded by bsz=1")
    print(f"    So we should have:")
    if site_idx > 0:
        left_sites = list(range(site_idx))
        print(f"      LEFT: sites {left_sites} → should be contracted, but bond to site {site_idx} preserved")
    else:
        print(f"      LEFT: empty (we're at leftmost site)")
    
    if site_idx < L - 1:
        right_sites = list(range(site_idx + 1, L))
        print(f"      RIGHT: sites {right_sites} → should be contracted, but bond to site {site_idx} preserved")
    else:
        print(f"      RIGHT: empty (we're at rightmost site)")
    
    print(f"      EXCLUDED: site {site_idx} (MPS_{site_idx} and INPUT_{site_idx})")
    
    # Check what tags are present
    all_tags = set()
    for t in current_env.tensors:
        all_tags.update(t.tags)
    
    print(f"\n  All tags in env(): {sorted([tag for tag in all_tags if not tag.startswith('I')])}")
    
    # Check if target is in environment
    target_in_env = f"MPS_{site_idx}" in all_tags
    input_in_env = f"INPUT_{site_idx}" in all_tags
    
    print(f"\n  MPS_{site_idx} in environment: {'YES ❌ (should be excluded!)' if target_in_env else 'NO ✓'}")
    print(f"  INPUT_{site_idx} in environment: {'YES ❌ (should be excluded!)' if input_in_env else 'NO ✓'}")
    
    # Check what OTHER MPS sites are present
    other_mps = [f"MPS_{j}" for j in range(L) if j != site_idx]
    present_mps = [tag for tag in other_mps if tag in all_tags]
    missing_mps = [tag for tag in other_mps if tag not in all_tags]
    
    print(f"\n  Other MPS sites present: {present_mps}")
    print(f"  Other MPS sites missing: {missing_mps}")
    if missing_mps:
        print(f"    ⚠️  Some MPS sites are missing - they were contracted!")
    
    # Get outer indices
    outer_inds = current_env.outer_inds()
    print(f"\n  Outer indices: {outer_inds}")
    
    # What are the MPS bond indices at this site?
    target = psi[f"I{site_idx}"]
    target_bonds = [idx for idx in target.inds if idx.startswith('_')]
    print(f"\n  MPS_{site_idx} bond indices: {target_bonds}")
    
    bonds_in_outer = [b for b in target_bonds if b in outer_inds]
    bonds_missing = [b for b in target_bonds if b not in outer_inds]
    
    print(f"  Bonds present in outer_inds: {bonds_in_outer} {'✓' if len(bonds_in_outer) == len(target_bonds) else '❌'}")
    if bonds_missing:
        print(f"  Bonds MISSING from outer_inds: {bonds_missing} ❌")
        print(f"    → These bonds were consumed during contraction!")

print("\n" + "="*80)
print("SUMMARY")
print("="*80)

print("""
KEY QUESTION: What does env() return?

From the test above, we can see:
1. env() returns a TensorNetwork at position i
2. It SHOULD exclude tensors at site i (bsz=1)
3. It SHOULD include LEFT and RIGHT environments
4. LEFT/RIGHT are PRE-CONTRACTED to be efficient

THE PROBLEM:
- When contracting LEFT/RIGHT, the MPS bond indices are being consumed
- The bond between site i and site i+1 disappears during contraction
- This means outer_inds() doesn't include the MPS bonds we need!

THE ISSUE IS IN: _get_inds_to_keep() or init_segment()
- These methods are contracting too aggressively
- They need to PRESERVE the bond indices that connect to the excluded site
""")
