"""
Test naming/labeling consistency in BatchMovingEnvironment
Focus: Verify that index names, tags, and tensor references are consistent
"""
import sys
sys.path.insert(0, 'model')

import quimb.tensor as qb
import torch
from batch_moving_environment import BatchMovingEnvironment

qb.set_tensor_linop_backend('torch')

def print_section(title):
    print("\n" + "="*80)
    print(f"  {title}")
    print("="*80 + "\n")

print_section("NAMING CONSISTENCY TEST FOR BatchMovingEnvironment")

# Setup
L, r, p = 4, 3, 2
BATCH = 10  # Small batch for easier debugging

print(f"Configuration:")
print(f"  L (sites): {L}")
print(f"  r (bond_dim): {r}")
print(f"  p (phys_dim): {p}")
print(f"  BATCH: {BATCH}")

# Create MPS states with explicit naming
psi = qb.MPS_rand_state(L, bond_dim=r, phys_dim=p)
phi = qb.MPS_rand_state(L, bond_dim=r, phys_dim=p)

psi.apply_to_arrays(lambda x: torch.tensor(x, dtype=torch.float32))
phi.apply_to_arrays(lambda x: torch.tensor(x, dtype=torch.float32))

# Document original indices
print_section("1. ORIGINAL MPS INDICES")
print("PSI original physical indices (before reindex):")
for i in range(L):
    t = psi[f"I{i}"]
    print(f"  Site {i}: indices={t.inds}, tags={t.tags}")

print("\nPHI original physical indices (before reindex):")
for i in range(L):
    t = phi[f"I{i}"]
    print(f"  Site {i}: indices={t.inds}, tags={t.tags}")

# Reindex to avoid conflicts - CRITICAL NAMING STEP
psi.reindex({f"k{i}": f"psi_phys_{i}" for i in range(L)}, inplace=True)
phi.reindex({f"k{i}": f"phi_phys_{i}" for i in range(L)}, inplace=True)

print_section("2. AFTER REINDEXING")
print("PSI physical indices after reindex:")
for i in range(L):
    t = psi[f"I{i}"]
    print(f"  Site {i}: indices={t.inds}")

print("\nPHI physical indices after reindex:")
for i in range(L):
    t = phi[f"I{i}"]
    print(f"  Site {i}: indices={t.inds}")

# Add unique tags to each tensor
print_section("3. ADDING UNIQUE TAGS")
for i in range(L):
    psi.add_tag(f"PSI_BLOCK_{i}", where=f"I{i}")
    phi.add_tag(f"PHI_BLOCK_{i}", where=f"I{i}")

print("PSI tags after adding unique tags:")
for i in range(L):
    t = psi[f"I{i}"]
    print(f"  Site {i}: tags={t.tags}")

print("\nPHI tags after adding unique tags:")
for i in range(L):
    t = phi[f"I{i}"]
    print(f"  Site {i}: tags={t.tags}")

# Create input tensors with consistent naming
print_section("4. INPUT TENSORS (BATCH DIMENSION)")
inputs = [
    qb.Tensor(
        data=torch.rand(BATCH, p, p, dtype=torch.float32), 
        inds=['s', f'psi_phys_{i}', f'phi_phys_{i}'],  # CRITICAL: Must match reindexed names
        tags={f'I{i}', 'INPUT', f'INPUT_{i}'} 
    ) 
    for i in range(L)
]

print("Input tensors:")
for i, inp in enumerate(inputs):
    print(f"  Site {i}:")
    print(f"    indices: {inp.inds}")
    print(f"    tags: {inp.tags}")
    print(f"    shape: {inp.shape}")
    print(f"    Expected: ('s', 'psi_phys_{i}', 'phi_phys_{i}') -> ({BATCH}, {p}, {p})")

# Build full tensor network
print_section("5. FULL TENSOR NETWORK")
tn_overlap = psi & phi
for t in inputs:
    tn_overlap.add_tensor(t)

print(f"Full TN has {tn_overlap.num_tensors} tensors")
print(f"Expected: {L} (psi) + {L} (phi) + {L} (inputs) = {3*L}")

# Check all indices in the network
all_inds = set()
for t in tn_overlap.tensors:
    all_inds.update(t.inds)

print(f"\nAll unique indices in TN:")
for idx in sorted(all_inds):
    print(f"  {idx}")

# Verify batch index is present
if 's' in all_inds:
    print(f"\n✓ Batch index 's' found in TN")
else:
    print(f"\n✗ ERROR: Batch index 's' NOT found in TN!")

# Check for expected physical indices
expected_psi_phys = [f'psi_phys_{i}' for i in range(L)]
expected_phi_phys = [f'phi_phys_{i}' for i in range(L)]

missing_psi = [idx for idx in expected_psi_phys if idx not in all_inds]
missing_phi = [idx for idx in expected_phi_phys if idx not in all_inds]

if missing_psi:
    print(f"✗ ERROR: Missing PSI physical indices: {missing_psi}")
else:
    print(f"✓ All PSI physical indices present")

if missing_phi:
    print(f"✗ ERROR: Missing PHI physical indices: {missing_phi}")
else:
    print(f"✓ All PHI physical indices present")

# Initialize BatchMovingEnvironment
print_section("6. BatchMovingEnvironment INITIALIZATION")
env = BatchMovingEnvironment(tn_overlap, begin='left', bsz=1, batch_inds=['s'])

print(f"Environment initialized:")
print(f"  batch_inds: {env.batch_inds}")
print(f"  position: {env.pos}")
print(f"  bsz: {env.bsz}")

# Check environment at each position
print_section("7. ENVIRONMENT INDICES AT EACH POSITION")

for site_idx in range(L):
    env.move_to(site_idx)
    current_env = env()
    
    print(f"\nSite {site_idx}:")
    print(f"  Environment has {current_env.num_tensors} tensors")
    
    # Get all indices in environment
    env_inds = set()
    for t in current_env.tensors:
        env_inds.update(t.inds)
    
    print(f"  Indices in environment: {sorted(env_inds)}")
    
    # Check if batch index is present
    if 's' in env_inds:
        print(f"  ✓ Batch index 's' present")
    else:
        print(f"  ✗ Batch index 's' MISSING")
    
    # Get outer indices (bonds to excluded site)
    outer_inds = current_env.outer_inds()
    print(f"  Outer indices (bonds): {outer_inds}")

# Test hole creation and index matching
print_section("8. HOLE CREATION AND INDEX MATCHING")

for site_idx in range(L):
    env.move_to(site_idx)
    current_env = env()
    
    # Get target tensor
    target = psi[f"I{site_idx}"]
    target_tag = f"PSI_BLOCK_{site_idx}"
    
    print(f"\nSite {site_idx}:")
    print(f"  Target tag: {target_tag}")
    print(f"  Target indices: {target.inds}")
    
    # Create hole
    env_with_hole = current_env.copy()
    
    # Check if target is in environment
    if target_tag in env_with_hole.tags:
        env_with_hole.delete(target_tag)
        print(f"  ✓ Target found and deleted from environment")
    else:
        print(f"  ✗ Target tag '{target_tag}' NOT in environment")
        print(f"    Available tags: {sorted(env_with_hole.tags)}")
        continue
    
    # Get outer indices
    outer_inds = set(env_with_hole.outer_inds())
    target_inds = set(target.inds)
    
    # Find overlapping indices (bonds)
    bond_inds = outer_inds & target_inds
    
    print(f"  Environment outer indices: {outer_inds}")
    print(f"  Target indices: {target_inds}")
    print(f"  Overlapping (bond) indices: {bond_inds}")
    
    if len(bond_inds) > 0:
        print(f"  ✓ Found {len(bond_inds)} bond(s) connecting environment to target")
    else:
        print(f"  ✗ ERROR: NO bonds found!")
    
    # Check for batch index in environment
    env_all_inds = set()
    for t in env_with_hole.tensors:
        env_all_inds.update(t.inds)
    
    if 's' in env_all_inds:
        print(f"  ✓ Batch index 's' in environment")
        
        # Prepare indices to keep for contraction
        inds_to_keep = list(outer_inds)
        if 's' not in inds_to_keep:
            inds_to_keep.append('s')
        
        print(f"  Indices to keep in contraction: {inds_to_keep}")
        
        # Try contraction
        try:
            contracted = env_with_hole.contract(all, output_inds=inds_to_keep)
            print(f"  ✓ Contraction successful")
            print(f"    Result shape: {contracted.shape}")
            
            if hasattr(contracted, 'inds'):
                print(f"    Result indices: {contracted.inds}")
                if 's' in contracted.inds:
                    print(f"    ✓✓ Batch dimension preserved in result")
                else:
                    print(f"    ✗✗ Batch dimension LOST in result!")
            
        except Exception as e:
            print(f"  ✗ Contraction FAILED: {e}")
    else:
        print(f"  ✗ Batch index 's' NOT in environment")

# Test index renaming consistency
print_section("9. INDEX NAMING VERIFICATION")

print("Checking that physical indices were renamed correctly:")
for i in range(L):
    psi_tensor = psi[f"I{i}"]
    phi_tensor = phi[f"I{i}"]
    input_tensor = inputs[i]
    
    psi_phys = [idx for idx in psi_tensor.inds if 'psi_phys' in idx]
    phi_phys = [idx for idx in phi_tensor.inds if 'phi_phys' in idx]
    input_psi_phys = [idx for idx in input_tensor.inds if 'psi_phys' in idx]
    input_phi_phys = [idx for idx in input_tensor.inds if 'phi_phys' in idx]
    
    print(f"\nSite {i}:")
    print(f"  PSI physical: {psi_phys}")
    print(f"  PHI physical: {phi_phys}")
    print(f"  INPUT psi connection: {input_psi_phys}")
    print(f"  INPUT phi connection: {input_phi_phys}")
    
    # Verify matches
    expected_psi = f'psi_phys_{i}'
    expected_phi = f'phi_phys_{i}'
    
    matches = True
    if expected_psi not in psi_tensor.inds:
        print(f"  ✗ PSI missing expected index '{expected_psi}'")
        matches = False
    if expected_phi not in phi_tensor.inds:
        print(f"  ✗ PHI missing expected index '{expected_phi}'")
        matches = False
    if expected_psi not in input_tensor.inds:
        print(f"  ✗ INPUT missing PSI connection '{expected_psi}'")
        matches = False
    if expected_phi not in input_tensor.inds:
        print(f"  ✗ INPUT missing PHI connection '{expected_phi}'")
        matches = False
    
    if matches:
        print(f"  ✓✓ All indices match correctly!")

# Test tag naming consistency
print_section("10. TAG NAMING VERIFICATION")

print("Expected tags per site:")
for i in range(L):
    print(f"\nSite {i}:")
    
    # Get all tensors at this site
    site_tensors = tn_overlap.select(f"I{i}")
    
    print(f"  Number of tensors with tag 'I{i}': {site_tensors.num_tensors}")
    print(f"  Expected: 3 (PSI, PHI, INPUT)")
    
    # List all unique tags at this site
    site_tags = set()
    for t in site_tensors.tensors:
        site_tags.update(t.tags)
    
    print(f"  All tags at site: {sorted(site_tags)}")
    
    # Check for expected tags
    expected_tags = {f'I{i}', f'PSI_BLOCK_{i}', f'PHI_BLOCK_{i}', 'INPUT', f'INPUT_{i}'}
    
    if expected_tags.issubset(site_tags):
        print(f"  ✓ All expected tags present")
    else:
        missing = expected_tags - site_tags
        print(f"  ✗ Missing tags: {missing}")

print_section("SUMMARY")

print("""
Key naming conventions that MUST be consistent:

1. BATCH INDEX: 's'
   - Used in input tensors
   - Specified in batch_inds=['s']
   - Must be preserved in contractions

2. PHYSICAL INDICES:
   - PSI: 'psi_phys_0', 'psi_phys_1', ..., 'psi_phys_{L-1}'
   - PHI: 'phi_phys_0', 'phi_phys_1', ..., 'phi_phys_{L-1}'
   - INPUT must connect to both PSI and PHI physical indices

3. SITE TAGS:
   - All tensors at site i have tag 'I{i}'
   - This is used by MovingEnvironment.site_tag(i)

4. UNIQUE IDENTIFICATION TAGS:
   - PSI tensors: 'PSI_BLOCK_0', 'PSI_BLOCK_1', ...
   - PHI tensors: 'PHI_BLOCK_0', 'PHI_BLOCK_1', ...
   - INPUT tensors: 'INPUT_0', 'INPUT_1', ...
   - Used for env.delete(tag) operations

5. BOND INDICES:
   - Automatically generated by quimb
   - Connect adjacent MPS tensors
   - Should appear in environment's outer_inds()
""")

print("\n" + "="*80)
print("  TEST COMPLETE")
print("="*80)
