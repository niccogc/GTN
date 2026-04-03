# type: ignore
"""
Final test: Verify CPDA environment computation works with explicit output_inds.
"""

import torch
import quimb.tensor as qt

# CPDA setup
L = 3
rank = 2
phys_dim = 3
output_dim = 2
batch_size = 4
output_site = L - 1

print("=" * 60)
print("SETUP")
print("=" * 60)

# Create nodes
tensors = []
for i in range(L):
    if i == output_site:
        shape = (phys_dim, rank, output_dim)
        inds = (f"x{i}", "r", "out")
    else:
        shape = (phys_dim, rank)
        inds = (f"x{i}", "r")

    data = torch.randn(*shape) * 0.1
    tensor = qt.Tensor(data=data, inds=inds, tags={f"Node{i}"})
    tensors.append(tensor)
    print(f"Node{i}: shape={shape}, inds={inds}")

tn = qt.TensorNetwork(tensors)

# Create inputs
inputs = []
for i in range(L):
    data = torch.randn(batch_size, phys_dim)
    tensor = qt.Tensor(data=data, inds=("s", f"x{i}"), tags={f"Input{i}"})
    inputs.append(tensor)

batch_dim = "s"
output_dims = ["out"]

print("\n" + "=" * 60)
print("TEST: CPDA Environment with Explicit output_inds")
print("=" * 60)


def cpda_batch_environment(tn, inputs, target_tag, batch_dim, output_site, L):
    """
    Compute CPDA environment using target node's indices as output_inds.

    For NTN, we need env shape to match node indices + batch.
    - Non-output node: env has (s, x{i}, r, out)
    - Output node: env has (s, x{i}, r)
    """
    target_tensor = tn[target_tag]
    target_idx = int(target_tag.replace("Node", ""))
    is_output_node = target_idx == output_site

    # Create env_tn = tn + inputs, remove target
    env_tn = tn.copy() & inputs
    env_tn.delete(target_tag)

    # Build output_inds: batch + node's indices
    # For non-output nodes, also need 'out' (from output node in env)
    node_inds = list(target_tensor.inds)

    if is_output_node:
        # Output node has (x{i}, r, out) - env should have (s, x{i}, r)
        env_output_inds = [batch_dim] + [ind for ind in node_inds if ind != "out"]
    else:
        # Non-output node has (x{i}, r) - env should have (s, x{i}, r, out)
        env_output_inds = [batch_dim] + node_inds + ["out"]

    print(f"\n{target_tag} (is_output={is_output_node}):")
    print(f"  Node inds: {node_inds}")
    print(f"  Requesting env output_inds: {env_output_inds}")

    # Contract
    env_tensor = env_tn.contract(output_inds=env_output_inds)
    print(f"  Env result: shape={env_tensor.shape}, inds={env_tensor.inds}")

    return env_tensor


def verify_forward_from_env(env, target_tensor, batch_dim, output_dims, expected_shape):
    """Verify that env @ node produces correct output."""
    forward_tn = env & target_tensor
    result = forward_tn.contract(output_inds=[batch_dim] + output_dims)

    print(f"  Forward: shape={result.shape}, inds={result.inds}")

    if result.shape == expected_shape:
        print(f"  PASS!")
        return True
    else:
        print(f"  FAIL: expected {expected_shape}")
        return False


# Test each node
all_passed = True
expected_output_shape = (batch_size, output_dim)

for i in range(L):
    target_tag = f"Node{i}"
    env = cpda_batch_environment(tn, inputs, target_tag, batch_dim, output_site, L)
    target_tensor = tn[target_tag]
    passed = verify_forward_from_env(
        env, target_tensor, batch_dim, output_dims, expected_output_shape
    )
    all_passed &= passed

print("\n" + "=" * 60)
print("COMPARING WITH FULL FORWARD PASS")
print("=" * 60)

# Full forward
full_tn = tn & inputs
y_full = full_tn.contract(output_inds=[batch_dim] + output_dims)
print(f"Full forward: shape={y_full.shape}, inds={y_full.inds}")

# Forward from each environment should give same result
for i in range(L):
    target_tag = f"Node{i}"
    env = cpda_batch_environment(tn, inputs, target_tag, batch_dim, output_site, L)
    target_tensor = tn[target_tag]

    forward_tn = env & target_tensor
    y_from_env = forward_tn.contract(output_inds=[batch_dim] + output_dims)

    # Check if results match (use reasonable tolerance for float64)
    diff = torch.abs(y_full.data - y_from_env.data).max().item()
    print(f"  {target_tag}: max diff from full forward = {diff:.2e}", end="")
    if diff < 1e-6:
        print(" OK")
    else:
        print(" MISMATCH!")
        all_passed = False

print("\n" + "=" * 60)
print("GRADIENT COMPUTATION TEST")
print("=" * 60)

# Simulate NTN gradient computation for Node0
target_tag = "Node0"
target_tensor = tn[target_tag]
env = cpda_batch_environment(tn, inputs, target_tag, batch_dim, output_site, L)

print(f"\nGradient test for {target_tag}:")
print(f"  Target tensor: shape={target_tensor.shape}, inds={target_tensor.inds}")
print(f"  Environment: shape={env.shape}, inds={env.inds}")

# Simulate dL/dy: (batch, out)
dL_dy = qt.Tensor(data=torch.randn(batch_size, output_dim), inds=(batch_dim, "out"))
print(f"  dL/dy: shape={dL_dy.shape}, inds={dL_dy.inds}")

# Gradient: contract env with dL_dy, output should have node's indices
node_inds = target_tensor.inds
grad_tn = env & dL_dy
node_grad = grad_tn.contract(output_inds=list(node_inds))

print(f"  Gradient: shape={node_grad.shape}, inds={node_grad.inds}")
print(f"  Expected: shape={target_tensor.shape}, inds={target_tensor.inds}")

if node_grad.shape == target_tensor.shape and set(node_grad.inds) == set(target_tensor.inds):
    print("  PASS!")
else:
    print("  FAIL!")
    all_passed = False

print("\n" + "=" * 60)
if all_passed:
    print("ALL TESTS PASSED!")
    print("\nCPDA CAN work with standard NTN if we modify _batch_environment to:")
    print("  1. Use node's own indices as output_inds (not just outer_inds)")
    print("  2. For non-output nodes: include 'out' in output_inds")
    print("  3. For output node: exclude 'out' from output_inds (it comes from node)")
else:
    print("SOME TESTS FAILED!")
print("=" * 60)
