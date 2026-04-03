# type: ignore
"""
Test unified environment output_inds logic that works for both MPO2 and CPDA.

Formula:
  env_output_inds = batch_dim ∪ node_inds ∪ out_labels - (node_inds ∩ out_labels)

This simplifies to:
  - If node has NO 'out': batch_dim ∪ node_inds ∪ out_labels
  - If node HAS 'out': batch_dim ∪ node_inds - {out}
"""

import torch
import quimb.tensor as qt
from model.standard import MPO2

# Parameters
L = 4
bond_dim = 3
phys_dim = 5
output_dim = 2
batch_size = 8
batch_dim = "s"
out_labels = {"out"}


def compute_env_output_inds(node_inds, batch_dim, out_labels):
    """
    Compute environment output_inds using the unified formula:

    env_inds = {batch_dim} ∪ node_inds ∪ out_labels - (node_inds ∩ out_labels)
    """
    node_inds_set = set(node_inds)
    intersection = node_inds_set & out_labels
    result = ({batch_dim} | node_inds_set | out_labels) - intersection
    return result


def create_inputs(L, batch_size, phys_dim):
    inputs = []
    for i in range(L):
        data = torch.randn(batch_size, phys_dim)
        tensor = qt.Tensor(data=data, inds=("s", f"x{i}"), tags={f"Input{i}"})
        inputs.append(tensor)
    return inputs


print("=" * 70)
print("TEST 1: MPO2 - Verify new logic matches existing outer_inds behavior")
print("=" * 70)

mpo2 = MPO2(L=L, bond_dim=bond_dim, phys_dim=phys_dim, output_dim=output_dim)
tn_mpo2 = mpo2.tn
inputs = create_inputs(L, batch_size, phys_dim)

print("\nMPO2 Tensors:")
for tensor in tn_mpo2:
    print(f"  {list(tensor.tags)[0]}: inds={tensor.inds}")

print("\nComparing env_output_inds:")
all_match = True

for i in range(L):
    target_tag = f"Node{i}"
    target_tensor = tn_mpo2[target_tag]
    node_inds = target_tensor.inds

    # Compute using new formula
    new_output_inds = compute_env_output_inds(node_inds, batch_dim, out_labels)

    # Get current outer_inds from env_tn
    env_tn = tn_mpo2.copy() & inputs
    env_tn.delete(target_tag)
    current_outer = set(env_tn.outer_inds())

    # Current NTN adds batch_dim if not in outer
    # (see lines 345-351 of NTN.py)
    all_inds_in_env = set()
    for t in env_tn:
        all_inds_in_env.update(t.inds)
    if batch_dim in all_inds_in_env:
        current_outer.add(batch_dim)

    match = new_output_inds == current_outer
    status = "✓" if match else "✗"
    all_match &= match

    print(f"\n  {target_tag}: node_inds={node_inds}")
    print(f"    New formula:    {sorted(new_output_inds)}")
    print(f"    Current outer:  {sorted(current_outer)}")
    print(f"    Match: {status}")

print("\n" + "=" * 70)
print("TEST 2: CPDA - Verify new logic gives correct indices")
print("=" * 70)

# Create CPDA TN manually
tensors_cpda = []
output_site = L - 1
for i in range(L):
    if i == output_site:
        shape = (phys_dim, bond_dim, output_dim)  # using bond_dim as rank
        inds = (f"x{i}", "r", "out")
    else:
        shape = (phys_dim, bond_dim)
        inds = (f"x{i}", "r")

    data = torch.randn(*shape) * 0.1
    tensor = qt.Tensor(data=data, inds=inds, tags={f"Node{i}"})
    tensors_cpda.append(tensor)

tn_cpda = qt.TensorNetwork(tensors_cpda)

print("\nCPDA Tensors:")
for tensor in tn_cpda:
    print(f"  {list(tensor.tags)[0]}: inds={tensor.inds}")

print("\nComputing env_output_inds with new formula:")
for i in range(L):
    target_tag = f"Node{i}"
    target_tensor = tn_cpda[target_tag]
    node_inds = target_tensor.inds

    new_output_inds = compute_env_output_inds(node_inds, batch_dim, out_labels)

    # What we expect:
    # - Non-output node (x{i}, r): should get {s, x{i}, r, out}
    # - Output node (x{i}, r, out): should get {s, x{i}, r}

    is_output_node = i == output_site
    if is_output_node:
        expected = {batch_dim, f"x{i}", "r"}
    else:
        expected = {batch_dim, f"x{i}", "r", "out"}

    match = new_output_inds == expected
    status = "✓" if match else "✗"
    all_match &= match

    print(f"\n  {target_tag} (output_node={is_output_node}): node_inds={node_inds}")
    print(f"    New formula: {sorted(new_output_inds)}")
    print(f"    Expected:    {sorted(expected)}")
    print(f"    Match: {status}")

print("\n" + "=" * 70)
print("TEST 3: CPDA - Verify environment contraction works with new indices")
print("=" * 70)

inputs_cpda = create_inputs(L, batch_size, phys_dim)

for i in range(L):
    target_tag = f"Node{i}"
    target_tensor = tn_cpda[target_tag]
    node_inds = target_tensor.inds

    # Compute output_inds
    output_inds = compute_env_output_inds(node_inds, batch_dim, out_labels)

    # Create and contract environment
    env_tn = tn_cpda.copy() & inputs_cpda
    env_tn.delete(target_tag)

    # Filter to only indices that exist in env_tn
    all_inds = set()
    for t in env_tn:
        all_inds.update(t.inds)
    valid_output_inds = [ind for ind in output_inds if ind in all_inds]

    try:
        env = env_tn.contract(output_inds=valid_output_inds)

        # Verify forward works
        forward_tn = env & target_tensor
        result = forward_tn.contract(output_inds=[batch_dim, "out"])

        expected_shape = (batch_size, output_dim)
        if result.shape == expected_shape:
            print(f"  {target_tag}: env shape={env.shape}, forward shape={result.shape} ✓")
        else:
            print(f"  {target_tag}: FAILED - forward shape {result.shape} != {expected_shape}")
            all_match = False
    except Exception as e:
        print(f"  {target_tag}: ERROR - {e}")
        all_match = False

print("\n" + "=" * 70)
if all_match:
    print("ALL TESTS PASSED!")
    print("\nThe unified formula works for both MPO2 and CPDA:")
    print("  env_inds = {batch_dim} ∪ node_inds ∪ out_labels - (node_inds ∩ out_labels)")
else:
    print("SOME TESTS FAILED!")
print("=" * 70)
