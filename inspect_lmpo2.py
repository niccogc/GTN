# type: ignore
"""
Inspect LMPO2 tensor labels and connections
"""
import torch
from model.MPO2_models import LMPO2

torch.set_default_dtype(torch.float32)

print("="*70)
print("LMPO2 TENSOR INSPECTION")
print("="*70)

# Create LMPO2 model
L = 3
bond_dim = 4
input_dim = 5
reduced_dim = 3
output_dim = 3

print(f"\nCreating LMPO2 with:")
print(f"  L = {L}")
print(f"  bond_dim = {bond_dim}")
print(f"  input_dim = {input_dim}")
print(f"  reduced_dim = {reduced_dim}")
print(f"  output_dim = {output_dim}")

lmpo2 = LMPO2(
    L=L,
    bond_dim=bond_dim,
    input_dim=input_dim,
    reduced_dim=reduced_dim,
    output_dim=output_dim,
    output_site=1
)

print(f"\n" + "="*70)
print("MODEL ATTRIBUTES")
print("="*70)
print(f"input_labels: {lmpo2.input_labels}")
print(f"input_dims: {lmpo2.input_dims}")
print(f"output_dims: {lmpo2.output_dims}")
print(f"reduction_factor: {lmpo2.reduction_factor:.2%}")

print(f"\n" + "="*70)
print("TENSOR NETWORK STRUCTURE")
print("="*70)
print(f"Total tensors: {len(lmpo2.tn.tensors)}")
print(f"Outer indices: {sorted(lmpo2.tn.outer_inds())}")
print(f"Inner indices: {sorted(lmpo2.tn.inner_inds())}")

print(f"\n" + "="*70)
print("MPO LAYER TENSORS (Dimensionality Reduction)")
print("="*70)
for i, tensor in enumerate(lmpo2.mpo_tensors):
    print(f"\nMPO Tensor {i}:")
    print(f"  Tags: {tensor.tags}")
    print(f"  Indices: {tensor.inds}")
    print(f"  Shape: {tensor.shape}")
    print(f"  Index sizes:")
    for idx in tensor.inds:
        print(f"    {idx}: {tensor.ind_size(idx)}")

print(f"\n" + "="*70)
print("MPS LAYER TENSORS (Output Generation)")
print("="*70)
for i, tensor in enumerate(lmpo2.mps_tensors):
    print(f"\nMPS Tensor {i}:")
    print(f"  Tags: {tensor.tags}")
    print(f"  Indices: {tensor.inds}")
    print(f"  Shape: {tensor.shape}")
    print(f"  Index sizes:")
    for idx in tensor.inds:
        print(f"    {idx}: {tensor.ind_size(idx)}")

print(f"\n" + "="*70)
print("CONNECTION ANALYSIS")
print("="*70)

print("\n1. Input Connections:")
print(f"   Expected input labels: {lmpo2.input_labels}")
print(f"   MPO physical indices: ", end="")
for i in range(L):
    mpo_input_idx = f"{i}_in"
    if mpo_input_idx in lmpo2.tn.outer_inds():
        print(f"{mpo_input_idx}(outer) ", end="")
    else:
        print(f"{mpo_input_idx}(MISSING!) ", end="")
print()

print("\n2. MPO → MPS Connections (via reduced indices):")
for i in range(L):
    reduced_idx = f"{i}_reduced"
    # Check if this index exists and if it's shared between MPO and MPS
    mpo_has = reduced_idx in lmpo2.mpo_tensors[i].inds
    mps_has = reduced_idx in lmpo2.mps_tensors[i].inds
    is_inner = reduced_idx in lmpo2.tn.inner_inds()
    
    print(f"   {reduced_idx}: MPO={mpo_has}, MPS={mps_has}, Inner={is_inner}")
    if mpo_has and mps_has and is_inner:
        mpo_size = lmpo2.mpo_tensors[i].ind_size(reduced_idx)
        mps_size = lmpo2.mps_tensors[i].ind_size(reduced_idx)
        match = "✓" if mpo_size == mps_size else "✗"
        print(f"            Size match {match}: MPO={mpo_size}, MPS={mps_size}")

print("\n3. Bond Connections:")
print("   MPO bonds:")
for i in range(L-1):
    bond_idx = f"b_mpo_{i}"
    if bond_idx in lmpo2.tn.inner_inds():
        print(f"     {bond_idx} (connects MPO {i} and {i+1})")
    else:
        print(f"     {bond_idx} MISSING!")

print("   MPS bonds:")
for i in range(L-1):
    bond_idx = f"b_mps_{i}"
    if bond_idx in lmpo2.tn.inner_inds():
        print(f"     {bond_idx} (connects MPS {i} and {i+1})")
    else:
        print(f"     {bond_idx} MISSING!")

print("\n4. Output Connection:")
output_idx = "out"
if output_idx in lmpo2.tn.outer_inds():
    # Find which tensor has the output
    for i, tensor in enumerate(lmpo2.mps_tensors):
        if output_idx in tensor.inds:
            output_size = tensor.ind_size(output_idx)
            print(f"   ✓ Output index 'out' found in MPS tensor {i} with size {output_size}")
            break
else:
    print(f"   ✗ Output index 'out' NOT FOUND in outer indices!")

print(f"\n" + "="*70)
print("VALIDATION")
print("="*70)

# Check that input labels match what builder will create
print("\n1. Input Label Match:")
print(f"   input_labels = {lmpo2.input_labels}")
print(f"   input_dims = {lmpo2.input_dims}")
if lmpo2.input_labels == lmpo2.input_dims:
    print(f"   ✓ Match! Builder will create inputs with these indices.")
else:
    print(f"   ✗ Mismatch! This could cause issues.")

# Check that all expected outer indices exist
print("\n2. Expected Outer Indices:")
expected_outer = set(lmpo2.input_dims + lmpo2.output_dims)
actual_outer = set(lmpo2.tn.outer_inds())
print(f"   Expected: {sorted(expected_outer)}")
print(f"   Actual:   {sorted(actual_outer)}")
if expected_outer == actual_outer:
    print(f"   ✓ Perfect match!")
else:
    missing = expected_outer - actual_outer
    extra = actual_outer - expected_outer
    if missing:
        print(f"   ✗ Missing: {missing}")
    if extra:
        print(f"   ✗ Extra: {extra}")

print("\n3. Layer Connection Check:")
all_connected = True
for i in range(L):
    reduced_idx = f"{i}_reduced"
    if reduced_idx not in lmpo2.tn.inner_inds():
        print(f"   ✗ {reduced_idx} is NOT an inner index (MPO-MPS disconnected!)")
        all_connected = False

if all_connected:
    print(f"   ✓ All MPO-MPS connections are inner indices (properly connected)")

print(f"\n" + "="*70)
