# type: ignore
"""
CRITICAL TEST: Check if _LEFT, _RIGHT, and SAME_SITE tensors 
INDIVIDUALLY have the correct bond indices.
"""
import sys
sys.path.insert(0, 'model')

import torch
import quimb.tensor as qt
from batch_moving_environment import BatchMovingEnvironment

qt.set_tensor_linop_backend('torch')

print("="*80)
print("TEST: _LEFT, _RIGHT, SAME_SITE INDIVIDUAL BOND INDICES")
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
bonds_map = {}
for i in range(L-1):
    t_i = psi[f"I{i}"]
    t_next = psi[f"I{i+1}"]
    shared = set(t_i.inds) & set(t_next.inds)
    if shared:
        bond = list(shared)[0]
        bonds_map[f"{i}→{i+1}"] = bond
        print(f"  Bond {i}→{i+1}: {bond}")

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
print("TESTING EACH SITE - LEFT TO RIGHT")
print("="*80)

for site_idx in range(L):
    print(f"\n{'─'*80}")
    print(f"SITE {site_idx}")
    print(f"{'─'*80}")
    
    env.move_to(site_idx)
    current_env = env()
    
    # Separate LEFT, RIGHT, SAME_SITE
    left_tensors = [t for t in current_env.tensors if '_LEFT' in t.tags]
    right_tensors = [t for t in current_env.tensors if '_RIGHT' in t.tags]
    site_tensors = [t for t in current_env.tensors if f'I{site_idx}' in t.tags and '_LEFT' not in t.tags and '_RIGHT' not in t.tags]
    
    print(f"\nEnvironment composition:")
    print(f"  LEFT: {len(left_tensors)} tensor(s)")
    print(f"  RIGHT: {len(right_tensors)} tensor(s)")
    print(f"  SAME_SITE: {len(site_tensors)} tensor(s)")
    
    # Expected bonds for this site
    expected_left_bond = bonds_map.get(f"{site_idx-1}→{site_idx}") if site_idx > 0 else None
    expected_right_bond = bonds_map.get(f"{site_idx}→{site_idx+1}") if site_idx < L-1 else None
    
    print(f"\nExpected bonds for site {site_idx}:")
    if expected_left_bond:
        print(f"  Left bond (from site {site_idx-1}): {expected_left_bond}")
    if expected_right_bond:
        print(f"  Right bond (to site {site_idx+1}): {expected_right_bond}")
    
    # Check LEFT tensor
    if left_tensors:
        print(f"\n_LEFT tensor:")
        for t in left_tensors:
            print(f"  Indices: {t.inds}")
            print(f"  Shape: {t.shape}")
            
            if expected_left_bond:
                if expected_left_bond in t.inds:
                    print(f"  ✓✓✓ HAS left bond {expected_left_bond}")
                else:
                    print(f"  ✗✗✗ MISSING left bond {expected_left_bond}")
            
            # Check batch
            if 's' in t.inds:
                print(f"  ✓ Has batch 's'")
            else:
                print(f"  ✗ Missing batch 's'")
    else:
        print(f"\n_LEFT: (empty)")
        if expected_left_bond:
            print(f"  ✗✗✗ Expected to have left bond {expected_left_bond}")
    
    # Check RIGHT tensor
    if right_tensors:
        print(f"\n_RIGHT tensor:")
        for t in right_tensors:
            print(f"  Indices: {t.inds}")
            print(f"  Shape: {t.shape}")
            
            if expected_right_bond:
                if expected_right_bond in t.inds:
                    print(f"  ✓✓✓ HAS right bond {expected_right_bond}")
                else:
                    print(f"  ✗✗✗ MISSING right bond {expected_right_bond}")
            
            # Check batch
            if 's' in t.inds:
                print(f"  ✓ Has batch 's'")
            else:
                print(f"  ✗ Missing batch 's'")
    else:
        print(f"\n_RIGHT: (empty)")
        if expected_right_bond:
            print(f"  ✗✗✗ Expected to have right bond {expected_right_bond}")
    
    # Check SAME_SITE tensors
    print(f"\nSAME_SITE tensors:")
    for t in site_tensors:
        tags_str = ', '.join([tag for tag in t.tags if not tag.startswith('I')])
        print(f"  [{tags_str}]: {t.inds}, shape={t.shape}")
        
        # MPS tensor should have both bonds
        if f'MPS_{site_idx}' in t.tags:
            if expected_left_bond and expected_left_bond in t.inds:
                print(f"    ✓ Has left bond {expected_left_bond}")
            elif expected_left_bond:
                print(f"    ✗ Missing left bond {expected_left_bond}")
            
            if expected_right_bond and expected_right_bond in t.inds:
                print(f"    ✓ Has right bond {expected_right_bond}")
            elif expected_right_bond:
                print(f"    ✗ Missing right bond {expected_right_bond}")

print("\n" + "="*80)
print("TESTING RIGHT TO LEFT")
print("="*80)

for site_idx in reversed(range(L)):
    print(f"\n{'─'*80}")
    print(f"SITE {site_idx}")
    print(f"{'─'*80}")
    
    env.move_to(site_idx)
    current_env = env()
    
    left_tensors = [t for t in current_env.tensors if '_LEFT' in t.tags]
    right_tensors = [t for t in current_env.tensors if '_RIGHT' in t.tags]
    
    expected_left_bond = bonds_map.get(f"{site_idx-1}→{site_idx}") if site_idx > 0 else None
    expected_right_bond = bonds_map.get(f"{site_idx}→{site_idx+1}") if site_idx < L-1 else None
    
    # Quick check
    left_ok = True
    right_ok = True
    
    if left_tensors and expected_left_bond:
        if expected_left_bond not in left_tensors[0].inds:
            print(f"  ✗ _LEFT missing bond {expected_left_bond}")
            left_ok = False
    
    if right_tensors and expected_right_bond:
        if expected_right_bond not in right_tensors[0].inds:
            print(f"  ✗ _RIGHT missing bond {expected_right_bond}")
            right_ok = False
    
    if left_ok and right_ok:
        print(f"  ✓ All bonds present")

print("\n" + "="*80)
print("SUMMARY")
print("="*80)

print("""
CRITICAL CHECK: Do LEFT, RIGHT, SAME_SITE have the correct bonds?

For each site i:
  _LEFT should have:
    - Bond connecting to site i (left side)
    - Batch dimension 's'
  
  _RIGHT should have:
    - Bond connecting to site i (right side)
    - Batch dimension 's'
  
  SAME_SITE should have:
    - MPS_i with BOTH left and right bonds
    - INPUT_i with batch 's' and physical 'phys_i'

If _LEFT or _RIGHT are missing the bonds to site i, 
the fix is not working correctly!
""")
