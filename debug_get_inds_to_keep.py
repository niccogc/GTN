# type: ignore
"""
Debug the _get_inds_to_keep logic in init_segment.
Trace through exactly what happens when contracting RIGHT + site(i+bsz).
"""
import sys
sys.path.insert(0, 'model')

import torch
import quimb.tensor as qt

qt.set_tensor_linop_backend('torch')

print("="*80)
print("DEBUG _get_inds_to_keep IN init_segment")  
print("="*80)

# Simple MPS setup
L = 4
BOND_DIM = 3
PHYS_DIM = 2
BATCH_SIZE = 100

psi = qt.MPS_rand_state(L, bond_dim=BOND_DIM, phys_dim=PHYS_DIM)
psi.apply_to_arrays(lambda x: torch.tensor(x, dtype=torch.float32))
psi.reindex({f"k{i}": f"phys_{i}" for i in range(L)}, inplace=True)

for i in range(L):
    psi.add_tag(f"MPS_{i}", where=f"I{i}")

print("\nMPS structure:")
for i in range(L):
    t = psi[f"I{i}"]
    bonds = [idx for idx in t.inds if idx.startswith('_')]
    print(f"  MPS_{i}: bonds={bonds}, physical=phys_{i}")

# Find bond indices connecting sites
print("\nBond connections:")
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

print(f"\nTotal TN: {tn.num_tensors} tensors")

# Simulate init_segment
print("\n" + "="*80)
print("SIMULATING init_segment(begin='left', bsz=1)")
print("="*80)

start, stop, bsz = 0, 4, 1

# Create tnc
tnc = tn.copy(virtual=True)
tnc |= qt.Tensor(data=torch.tensor(1.0, dtype=torch.float32), tags="_LEFT")
tnc |= qt.Tensor(data=torch.tensor(1.0, dtype=torch.float32), tags="_RIGHT")

# Focus on building env[0]
print("\n" + "─"*80)
print("BUILDING envs[0] - THE CRITICAL CASE")
print("─"*80)

i = 0
print(f"\nAt site i={i}:")
print(f"  - We need to contract (_RIGHT, I{i+bsz}) = (_RIGHT, I1)")
print(f"  - This includes: _RIGHT boundary + MPS_1 + INPUT_1")
print(f"  - After contraction, we need bond {i}→{i+1} preserved")
print(f"  - That bond connects MPS_0 (at site {i}) to MPS_1 (being contracted)")

# Select tensors being contracted
tags_to_contract = ("_RIGHT", f"I{i + bsz}")
active_tn = tnc.select(tags_to_contract, which='any')

print(f"\nActive tensors (being contracted):")
for t in active_tn.tensors:
    tags_str = ', '.join([tag for tag in t.tags if not tag.startswith('I')])
    print(f"  [{tags_str}]: {t.inds}")

# Also show what's at site i (NOT being contracted)
site_i_tn = tnc.select(f"I{i}")
print(f"\nSite {i} tensors (NOT being contracted, passive):")
for t in site_i_tn.tensors:
    tags_str = ', '.join([tag for tag in t.tags if not tag.startswith('I')])
    print(f"  [{tags_str}]: {t.inds}")

# Compute indices
active_inds = set().union(*(t.inds for t in active_tn))
all_inds = set().union(*(t.inds for t in tnc))
passive_inds = all_inds - active_inds
site_i_inds = set().union(*(t.inds for t in site_i_tn))

print(f"\nIndex analysis:")
print(f"  Active indices (in tensors being contracted): {sorted(active_inds)}")
print(f"  Site {i} indices (passive): {sorted(site_i_inds)}")

# The critical bond
bond_0_to_1 = bonds_map.get(f"{i}→{i+1}")
print(f"\n  CRITICAL BOND {i}→{i+1}: {bond_0_to_1}")
print(f"    Is it in active? {bond_0_to_1 in active_inds}")
print(f"    Is it in site_{i}? {bond_0_to_1 in site_i_inds}")
print(f"    Is it in passive (all - active)? {bond_0_to_1 in passive_inds}")

# Current logic
bonds_current = active_inds & passive_inds
print(f"\n  Current _get_inds_to_keep logic:")
print(f"    bonds = active ∩ passive = {sorted(bonds_current)}")
print(f"    Does it include {bond_0_to_1}? {bond_0_to_1 in bonds_current}")

if bond_0_to_1 not in bonds_current:
    print(f"    ✗✗✗ BOND IS MISSING!")
    print(f"\n  Why?")
    print(f"    - Bond is in MPS_0 (at site {i})")
    print(f"    - Bond is also in MPS_1 (being contracted)")
    print(f"    - So bond is in active")
    print(f"    - But MPS_0 is at site {i}, not being contracted yet")
    print(f"    - However, passive = all - active")
    print(f"    - Since bond is in active, it's NOT in passive!")
    print(f"    - Therefore bonds = active ∩ passive doesn't include it")

# Proposed fix
print(f"\n" + "="*80)
print("PROPOSED FIX")
print("="*80)

print(f"""
The issue: bonds = active ∩ passive misses bonds to site {i}.

Current logic:
  - passive = all - active
  - bonds = active ∩ passive
  
This finds bonds between active tensors and OTHER passive tensors,
but NOT bonds to site {i} (which isn't in the contraction yet).

FIX: Also preserve bonds between active and site {i}:
  - site_{i}_inds = indices at site {i}
  - bonds_to_site_i = active ∩ site_{i}_inds
  - inds_to_keep = bonds | bonds_to_site_i | batch | output
  
This ensures the bond from site {i} to site {i+1} is preserved!
""")

# Test the fix
bonds_to_site_i = active_inds & site_i_inds
print(f"Testing the fix:")
print(f"  bonds_to_site_i = active ∩ site_{i}_inds = {sorted(bonds_to_site_i)}")
print(f"  Does it include {bond_0_to_1}? {bond_0_to_1 in bonds_to_site_i} ✓")

batch_in_active = {'s'} & active_inds
inds_to_keep_fixed = bonds_current | bonds_to_site_i | batch_in_active
print(f"\n  Final inds_to_keep (with fix): {sorted(inds_to_keep_fixed)}")
print(f"  Includes {bond_0_to_1}? {bond_0_to_1 in inds_to_keep_fixed} ✓")
print(f"  Includes 's'? {'s' in inds_to_keep_fixed} ✓")
