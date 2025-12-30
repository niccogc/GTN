# type: ignore
"""
Test BatchMovingEnvironment with output dimension on one of the MPS tensors.
Check if _LEFT, _RIGHT, SAME_SITE have correct bonds AND output dimension handling.
"""
import sys
sys.path.insert(0, 'model')

import torch
import quimb.tensor as qt
from batch_moving_environment import BatchMovingEnvironment

qt.set_tensor_linop_backend('torch')

print("="*80)
print("TEST: _LEFT, _RIGHT, SAME_SITE WITH OUTPUT DIMENSION")
print("="*80)

# Setup
L = 4
BOND_DIM = 3
PHYS_DIM = 2
BATCH_SIZE = 100
OUTPUT_DIM = 10

# Create MPS
psi = qt.MPS_rand_state(L, bond_dim=BOND_DIM, phys_dim=PHYS_DIM)

# Add output dimension to middle site BEFORE converting to torch
middle_site = 2
print(f"\nAdding 'out' dimension (size {OUTPUT_DIM}) to site {middle_site}")
middle_tensor = psi[f'I{middle_site}']
middle_tensor.new_ind('out', size=OUTPUT_DIM, axis=-1, mode='random', rand_strength=0.1)

# Now convert to torch
psi.apply_to_arrays(lambda x: torch.tensor(x, dtype=torch.float32))
psi.reindex({f"k{i}": f"phys_{i}" for i in range(L)}, inplace=True)

for i in range(L):
    psi.add_tag(f"MPS_{i}", where=f"I{i}")

print("\nMPS structure with output dimension:")
bonds_map = {}
for i in range(L):
    t = psi[f"I{i}"]
    bonds = [idx for idx in t.inds if idx.startswith('_')]
    has_out = 'out' in t.inds
    print(f"  MPS_{i}: bonds={bonds}, phys=phys_{i}{', OUT' if has_out else ''}")
    print(f"           indices={t.inds}, shape={t.shape}")

for i in range(L-1):
    t_i = psi[f"I{i}"]
    t_next = psi[f"I{i+1}"]
    shared = set(t_i.inds) & set(t_next.inds)
    if shared:
        bond = list(shared)[0]
        bonds_map[f"{i}→{i+1}"] = bond

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
env = BatchMovingEnvironment(tn, begin='left', bsz=1, batch_inds=['s'], output_dims={'out'})

print(f"\nBatchMovingEnvironment initialized:")
print(f"  batch_inds: {env.batch_inds}")
print(f"  output_dims: {env.output_dims}")

print("\n" + "="*80)
print("TESTING EACH SITE - LEFT TO RIGHT")
print("="*80)

for site_idx in range(L):
    print(f"\n{'─'*80}")
    print(f"SITE {site_idx}")
    print(f"{'─'*80}")
    
    env.move_to(site_idx)
    current_env = env()
    
    # Separate components
    left_tensors = [t for t in current_env.tensors if '_LEFT' in t.tags]
    right_tensors = [t for t in current_env.tensors if '_RIGHT' in t.tags]
    site_tensors = [t for t in current_env.tensors if f'I{site_idx}' in t.tags and '_LEFT' not in t.tags and '_RIGHT' not in t.tags]
    
    print(f"\nEnvironment composition:")
    print(f"  LEFT: {len(left_tensors)} tensor(s)")
    print(f"  RIGHT: {len(right_tensors)} tensor(s)")
    print(f"  SAME_SITE: {len(site_tensors)} tensor(s)")
    
    # Expected bonds
    expected_left_bond = bonds_map.get(f"{site_idx-1}→{site_idx}") if site_idx > 0 else None
    expected_right_bond = bonds_map.get(f"{site_idx}→{site_idx+1}") if site_idx < L-1 else None
    
    # Check if output dimension should be in environment
    site_has_output = (site_idx == middle_site)
    
    print(f"\nExpected at site {site_idx}:")
    if expected_left_bond:
        print(f"  Left bond: {expected_left_bond}")
    if expected_right_bond:
        print(f"  Right bond: {expected_right_bond}")
    print(f"  Site has 'out': {site_has_output}")
    
    # Check LEFT
    if left_tensors:
        print(f"\n_LEFT tensor:")
        t = left_tensors[0]
        print(f"  Indices: {t.inds}")
        print(f"  Shape: {t.shape}")
        
        if expected_left_bond:
            if expected_left_bond in t.inds:
                print(f"  ✓ Has left bond {expected_left_bond}")
            else:
                print(f"  ✗ MISSING left bond {expected_left_bond}")
        
        if 's' in t.inds:
            print(f"  ✓ Has batch 's'")
        else:
            print(f"  ✗ Missing batch 's'")
        
        # Check output dimension
        if site_idx > middle_site:
            # LEFT should have 'out' (it includes the middle site)
            if 'out' in t.inds:
                print(f"  ✓ Has 'out' (from site {middle_site})")
            else:
                print(f"  ✗ Missing 'out' (should have from site {middle_site})")
        elif site_idx <= middle_site:
            # LEFT should NOT have 'out'
            if 'out' not in t.inds:
                print(f"  ✓ No 'out' (correct)")
            else:
                print(f"  ✗ Has 'out' (shouldn't!)")
    else:
        print(f"\n_LEFT: (empty)")
    
    # Check RIGHT
    if right_tensors:
        print(f"\n_RIGHT tensor:")
        t = right_tensors[0]
        print(f"  Indices: {t.inds}")
        print(f"  Shape: {t.shape}")
        
        if expected_right_bond:
            if expected_right_bond in t.inds:
                print(f"  ✓ Has right bond {expected_right_bond}")
            else:
                print(f"  ✗ MISSING right bond {expected_right_bond}")
        
        if 's' in t.inds:
            print(f"  ✓ Has batch 's'")
        else:
            print(f"  ✗ Missing batch 's'")
        
        # Check output dimension
        if site_idx < middle_site:
            # RIGHT should have 'out' (it includes the middle site)
            if 'out' in t.inds:
                print(f"  ✓ Has 'out' (from site {middle_site})")
            else:
                print(f"  ✗ Missing 'out' (should have from site {middle_site})")
        elif site_idx >= middle_site:
            # RIGHT should NOT have 'out'
            if 'out' not in t.inds:
                print(f"  ✓ No 'out' (correct)")
            else:
                print(f"  ✗ Has 'out' (shouldn't!)")
    else:
        print(f"\n_RIGHT: (empty)")
    
    # Check SAME_SITE
    print(f"\nSAME_SITE tensors:")
    for t in site_tensors:
        tags_str = ', '.join([tag for tag in t.tags if not tag.startswith('I')])
        print(f"  [{tags_str}]: {t.inds}, shape={t.shape}")
        
        if f'MPS_{site_idx}' in t.tags:
            # Check bonds
            if expected_left_bond and expected_left_bond in t.inds:
                print(f"    ✓ Has left bond")
            elif expected_left_bond:
                print(f"    ✗ Missing left bond")
            
            if expected_right_bond and expected_right_bond in t.inds:
                print(f"    ✓ Has right bond")
            elif expected_right_bond:
                print(f"    ✗ Missing right bond")
            
            # Check output
            if site_has_output:
                if 'out' in t.inds:
                    print(f"    ✓ Has 'out' dimension")
                else:
                    print(f"    ✗ Missing 'out' dimension")

print("\n" + "="*80)
print("RIGHT TO LEFT SWEEP")
print("="*80)

for site_idx in reversed(range(L)):
    print(f"\nSite {site_idx}:", end=" ")
    
    env.move_to(site_idx)
    current_env = env()
    
    left_tensors = [t for t in current_env.tensors if '_LEFT' in t.tags]
    right_tensors = [t for t in current_env.tensors if '_RIGHT' in t.tags]
    
    expected_left_bond = bonds_map.get(f"{site_idx-1}→{site_idx}") if site_idx > 0 else None
    expected_right_bond = bonds_map.get(f"{site_idx}→{site_idx+1}") if site_idx < L-1 else None
    
    issues = []
    
    # Check bonds
    if left_tensors and expected_left_bond:
        if expected_left_bond not in left_tensors[0].inds:
            issues.append(f"LEFT missing {expected_left_bond}")
    
    if right_tensors and expected_right_bond:
        if expected_right_bond not in right_tensors[0].inds:
            issues.append(f"RIGHT missing {expected_right_bond}")
    
    # Check output dimension
    if site_idx > middle_site and left_tensors:
        if 'out' not in left_tensors[0].inds:
            issues.append("LEFT missing 'out'")
    
    if site_idx < middle_site and right_tensors:
        if 'out' not in right_tensors[0].inds:
            issues.append("RIGHT missing 'out'")
    
    if issues:
        print(f"✗ {', '.join(issues)}")
    else:
        print(f"✓ All correct")

print("\n" + "="*80)
print("SUMMARY")
print("="*80)

print(f"""
Output dimension is on site {middle_site}.

Expected behavior:
  - Sites 0, 1: _RIGHT should have 'out' (it includes site {middle_site})
  - Site {middle_site}: 'out' is on SAME_SITE MPS tensor, NOT in _LEFT or _RIGHT
  - Site 3: _LEFT should have 'out' (it includes site {middle_site})

The 'out' dimension should be treated like batch 's':
  - Preserved during contractions
  - Appears in LEFT/RIGHT when they include the site with 'out'
  - Does NOT appear in outer_inds of the site that has it (not connected to anything)
""")

print("\n" + "="*80)
print("TEST COMPLETE")
print("="*80)
