# type: ignore
"""
Test if LEFT, RIGHT, and SAME_SITE tensor networks have the correct indices
at each position, going both left and right.
"""
import sys
sys.path.insert(0, 'model')

import torch
import quimb.tensor as qt
from batch_moving_environment import BatchMovingEnvironment

qt.set_tensor_linop_backend('torch')

print("="*80)
print("TEST LEFT, RIGHT, SAME_SITE INDICES - ALL SITES, BOTH DIRECTIONS")
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

print("\nMPS structure:")
bond_map = {}
for i in range(L):
    t = psi[f"I{i}"]
    bonds = [idx for idx in t.inds if idx.startswith('_')]
    bond_map[i] = bonds
    print(f"  MPS_{i}: bonds={bonds}, physical=phys_{i}")

# Identify left/right bonds for each site
print("\nBond connections:")
for i in range(L):
    left_bond = None
    right_bond = None
    
    if i > 0:
        # Find bond shared with previous site
        shared = set(bond_map[i]) & set(bond_map[i-1])
        if shared:
            left_bond = list(shared)[0]
    
    if i < L - 1:
        # Find bond shared with next site
        shared = set(bond_map[i]) & set(bond_map[i+1])
        if shared:
            right_bond = list(shared)[0]
    
    print(f"  MPS_{i}: left_bond={left_bond}, right_bond={right_bond}")

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
print("LEFT TO RIGHT SWEEP")
print("="*80)

for site_idx in range(L):
    print(f"\n{'─'*80}")
    print(f"SITE {site_idx}")
    print(f"{'─'*80}")
    
    env.move_to(site_idx)
    current_env = env()
    
    # Identify LEFT, RIGHT, SAME_SITE tensors
    left_tensors = []
    right_tensors = []
    same_site_tensors = []
    
    for t in current_env.tensors:
        tags = set(t.tags)
        if '_LEFT' in tags:
            left_tensors.append(t)
        elif '_RIGHT' in tags:
            right_tensors.append(t)
        elif f'I{site_idx}' in tags:
            same_site_tensors.append(t)
    
    print(f"\nEnvironment composition:")
    print(f"  LEFT tensors: {len(left_tensors)}")
    print(f"  RIGHT tensors: {len(right_tensors)}")
    print(f"  SAME_SITE tensors: {len(same_site_tensors)}")
    
    # Analyze LEFT
    if left_tensors:
        print(f"\n  LEFT component:")
        for t in left_tensors:
            other_tags = [tag for tag in t.tags if tag not in ['_LEFT', f'I{site_idx}']]
            print(f"    Tags: {other_tags}")
            print(f"    Indices: {t.inds}")
            print(f"    Shape: {t.shape}")
            
            # Check if it has the bond to current site
            if site_idx > 0:
                shared = set(bond_map[site_idx]) & set(bond_map[site_idx-1])
                expected_bond = list(shared)[0] if shared else None
                if expected_bond and expected_bond in t.inds:
                    print(f"    ✓ Has bond {expected_bond} to MPS_{site_idx}")
                elif expected_bond:
                    print(f"    ✗ Missing bond {expected_bond} to MPS_{site_idx}")
    else:
        print(f"\n  LEFT: (empty)")
    
    # Analyze RIGHT
    if right_tensors:
        print(f"\n  RIGHT component:")
        for t in right_tensors:
            other_tags = [tag for tag in t.tags if tag not in ['_RIGHT', f'I{site_idx}']]
            print(f"    Tags: {other_tags}")
            print(f"    Indices: {t.inds}")
            print(f"    Shape: {t.shape}")
            
            # Check if it has the bond to current site
            if site_idx < L - 1:
                shared = set(bond_map[site_idx]) & set(bond_map[site_idx+1])
                expected_bond = list(shared)[0] if shared else None
                if expected_bond and expected_bond in t.inds:
                    print(f"    ✓ Has bond {expected_bond} to MPS_{site_idx}")
                elif expected_bond:
                    print(f"    ✗ Missing bond {expected_bond} to MPS_{site_idx}")
    else:
        print(f"\n  RIGHT: (empty)")
    
    # Analyze SAME_SITE
    print(f"\n  SAME_SITE tensors ({len(same_site_tensors)}):")
    for t in same_site_tensors:
        other_tags = [tag for tag in t.tags if not tag.startswith('I')]
        print(f"    Tags: {other_tags}")
        print(f"    Indices: {t.inds}")
        print(f"    Shape: {t.shape}")
    
    # Get all indices in environment
    all_env_inds = set()
    for t in current_env.tensors:
        all_env_inds.update(t.inds)
    
    print(f"\n  All indices in environment: {sorted(all_env_inds)}")
    
    # Check expected bonds
    target = psi[f"I{site_idx}"]
    target_bonds = [idx for idx in target.inds if idx.startswith('_')]
    
    print(f"\n  Target MPS_{site_idx} bonds: {target_bonds}")
    for bond in target_bonds:
        if bond in all_env_inds:
            print(f"    ✓ {bond} present in environment")
        else:
            print(f"    ✗ {bond} MISSING from environment")

print("\n" + "="*80)
print("RIGHT TO LEFT SWEEP")
print("="*80)

for site_idx in reversed(range(L)):
    print(f"\n{'─'*80}")
    print(f"SITE {site_idx}")
    print(f"{'─'*80}")
    
    env.move_to(site_idx)
    current_env = env()
    
    # Get all indices in environment
    all_env_inds = set()
    for t in current_env.tensors:
        all_env_inds.update(t.inds)
    
    # Check expected bonds
    target = psi[f"I{site_idx}"]
    target_bonds = [idx for idx in target.inds if idx.startswith('_')]
    
    print(f"  Target MPS_{site_idx} bonds: {target_bonds}")
    for bond in target_bonds:
        if bond in all_env_inds:
            print(f"    ✓ {bond} present in environment")
        else:
            print(f"    ✗ {bond} MISSING from environment")

print("\n" + "="*80)
print("SUMMARY")
print("="*80)

print("""
For BatchMovingEnvironment to work correctly:

At each site i, env() should return LEFT + RIGHT + SAME_SITE where:
  - LEFT: contracted environment of sites 0..i-1
    → Must preserve bond connecting to site i
  - RIGHT: contracted environment of sites i+1..L-1  
    → Must preserve bond connecting to site i
  - SAME_SITE: uncontracted tensors at site i (MPS_i + INPUT_i)

After deleting target from SAME_SITE, outer_inds() should include:
  - Bond from LEFT to site i (if i > 0)
  - Bond from RIGHT to site i (if i < L-1)
  - Physical index connecting to INPUT_i

If bonds are missing, the LEFT/RIGHT contractions are too aggressive.
""")
