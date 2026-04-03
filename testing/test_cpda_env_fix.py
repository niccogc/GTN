# type: ignore
"""
Test: Use target node's own indices as output_inds for environment contraction.
"""

import torch
import quimb.tensor as qt

# Simple CPDA: 3 nodes
L = 3
rank = 2
phys_dim = 3
output_dim = 2
batch_size = 4

print("=" * 60)
print("SETUP")
print("=" * 60)

# Create nodes
tensors = []
for i in range(L):
    if i == L - 1:  # output node
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
    print(f"Input{i}: shape={tensor.shape}, inds={tensor.inds}")

batch_dim = "s"
output_dims = ["out"]

print("\n" + "=" * 60)
print("TEST: Environment using node's own indices as output_inds")
print("=" * 60)


def compute_environment_with_node_inds(tn, inputs, target_tag, batch_dim, output_dims):
    """
    Compute environment using the target node's indices as output_inds.
    """
    target_tensor = tn[target_tag]
    node_inds = list(target_tensor.inds)

    print(f"\nTarget: {target_tag}")
    print(f"  Node inds: {node_inds}")

    # Create env_tn
    env_tn = tn.copy() & inputs
    env_tn.delete(target_tag)

    print(f"  env_tn outer_inds (default): {list(env_tn.outer_inds())}")

    # Build output_inds: batch + node's own indices
    # But we need to handle the case where node has 'out' index
    env_output_inds = [batch_dim] + node_inds

    print(f"  Requesting output_inds: {env_output_inds}")

    # Check which indices actually exist in env_tn
    all_inds_in_env = set()
    for t in env_tn:
        all_inds_in_env.update(t.inds)
    print(f"  All indices in env_tn: {all_inds_in_env}")

    # Filter to only indices that exist
    valid_output_inds = [ind for ind in env_output_inds if ind in all_inds_in_env]
    print(f"  Valid output_inds: {valid_output_inds}")

    # Contract
    try:
        env_tensor = env_tn.contract(output_inds=valid_output_inds)
        print(f"  Result: shape={env_tensor.shape}, inds={env_tensor.inds}")
        return env_tensor
    except Exception as e:
        print(f"  ERROR: {e}")
        return None


# Test for each node
for i in range(L):
    target_tag = f"Node{i}"
    env = compute_environment_with_node_inds(tn, inputs, target_tag, batch_dim, output_dims)

    if env is not None:
        # Verify: env @ node should give (batch, out)
        target_tensor = tn[target_tag]
        forward_tn = env & target_tensor

        # Output should be batch + output_dims (excluding node's own dims which get contracted)
        try:
            result = forward_tn.contract(output_inds=[batch_dim] + output_dims)
            print(f"  Forward check: shape={result.shape}, inds={result.inds}")
            expected_shape = (batch_size, output_dim)
            if result.shape == expected_shape:
                print(f"  PASS!")
            else:
                print(f"  FAIL: expected {expected_shape}")
        except Exception as e:
            print(f"  Forward ERROR: {e}")

print("\n" + "=" * 60)
print("ANALYSIS")
print("=" * 60)
print("""
The key insight:
- For non-output nodes (Node0, Node1): indices are (x{i}, r)
  Environment should have shape (batch, x{i}, r) to contract with node
  
- For output node (Node2): indices are (x{i}, r, out)  
  Environment should have shape (batch, x{i}, r) - 'out' comes from node itself!
  
When we request output_inds=[s, x{i}, r], quimb will:
1. Contract all other indices (x{j} for j != i)
2. Keep s, x{i}, r open

But 'r' is shared between remaining nodes... let's see if quimb handles this.
""")

print("\n" + "=" * 60)
print("TEST: What if 'r' is requested but shared?")
print("=" * 60)

# Manual test: create env_tn for Node0, request 'r' as output
target_tag = "Node0"
env_tn = tn.copy() & inputs
env_tn.delete(target_tag)

print("Tensors in env_tn:")
for t in env_tn:
    print(f"  {list(t.tags)}: inds={t.inds}")

print(f"\nRequesting output_inds=['s', 'x0', 'r']")
try:
    env = env_tn.contract(output_inds=["s", "x0", "r"])
    print(f"Result: shape={env.shape}, inds={env.inds}")
except Exception as e:
    print(f"ERROR: {e}")

print("\n--- The issue: 'r' appears in Node1 AND Node2 ---")
print("When both nodes have 'r', it's an internal index and gets contracted.")
print("We can't keep 'r' open because it connects multiple tensors.")

print("\n" + "=" * 60)
print("SOLUTION: Different index naming for CPDA?")
print("=" * 60)
print("""
Option 1: Use different rank indices per node: r0, r1, r2
  - Then add a "rank combiner" tensor that connects them
  - But this changes the CPDA structure...

Option 2: Custom environment computation for CPDA
  - Don't use quimb's contraction for environment
  - Manually compute: env_i = prod_{j!=i}(Input_j @ Node_j)
  - Keep 'r' dimension via elementwise operations

Option 3: Use per-node rank indices that are the SAME tensor
  - Node0: (x0, r), Node1: (x1, r), Node2: (x2, r, out)
  - For environment of Node0:
    - Contract Input1 with Node1 over x1 -> (s, r)
    - Contract Input2 with Node2 over x2 -> (s, r, out)
    - Elementwise multiply the (s, r) parts, keeping r open
    - Result: (s, r, out)
    - Then multiply by Input0 -> (s, x0, r, out)? No, Input0 doesn't have r...
    
Let me think again...
""")

print("\n" + "=" * 60)
print("CORRECT CPDA ENVIRONMENT MATH")
print("=" * 60)
print("""
Forward pass:
  y = sum_r [ prod_i (sum_{x_i} Input_i[s,x_i] * Node_i[x_i, r, ...]) ]
  
For Node0 with indices (x0, r):
  y = sum_r [ (sum_{x0} I0[s,x0] * N0[x0,r]) * (sum_{x1} I1[s,x1] * N1[x1,r]) * ... ]
  
Environment for Node0 = derivative of y w.r.t. Node0:
  dy/dN0[x0,r] = sum_{r'!=r} [...] + I0[s,x0] * prod_{i>0}(sum_{x_i} I_i[s,x_i] * N_i[x_i, r])
  
Wait, that's not right either. Let me be more careful.

Actually:
  y[s, out] = sum_r [ A0[s,r] * A1[s,r] * ... * A_{L-1}[s,r,out] ]
  
where A_i[s,r] = sum_{x_i} Input_i[s,x_i] * Node_i[x_i, r]  (for non-output nodes)
  and A_{L-1}[s,r,out] = sum_{x_{L-1}} Input_{L-1}[s,x_{L-1}] * Node_{L-1}[x_{L-1}, r, out]

The gradient w.r.t. Node_0[x0, r]:
  dy/dN0[x0, r] = Input_0[s, x0] * sum_{r} [ A1[s,r] * ... * A_{L-1}[s,r,out] ]
  
No wait, that sums over r, but we want to keep r for the gradient...

Let me reconsider. The output is:
  y[s, out] = sum_r prod_i A_i[s, r, ...]

To get gradient w.r.t N0[x0, r_fixed]:
  dy/dN0[x0, r_fixed] = I0[s, x0] * prod_{i>0} A_i[s, r_fixed, ...]
  
So the environment for Node0 is:
  env[s, x0, r, out] = I0[s, x0] * prod_{i>0} A_i[s, r, ...]
                     = I0[s, x0] * A1[s, r] * A2[s, r, out]
                     
This keeps 'r' open - no sum over r!
""")

print("\n" + "=" * 60)
print("IMPLEMENTING CORRECT CPDA ENVIRONMENT")
print("=" * 60)


def cpda_environment(tn, inputs, target_idx, L, batch_dim="s"):
    """
    Compute CPDA environment correctly by NOT summing over r.

    env[s, x_i, r, (out)] = Input_i[s, x_i] * prod_{j!=i} A_j[s, r, ...]
    where A_j = sum_{x_j} Input_j[s, x_j] * Node_j[x_j, r, ...]
    """
    target_input = inputs[target_idx]

    # Compute A_j for each j != target_idx
    A_list = []
    for j in range(L):
        if j == target_idx:
            continue
        node_j = tn[f"Node{j}"]
        input_j = inputs[j]
        # Contract over x_j only
        A_j_tn = node_j & input_j
        # The result should have indices: (s, r) or (s, r, out)
        A_j = A_j_tn.contract()
        print(f"  A_{j}: shape={A_j.shape}, inds={A_j.inds}")
        A_list.append(A_j)

    # Multiply all A_j together (elementwise over s and r)
    # Then multiply by target_input
    all_tensors = [target_input] + A_list
    env_tn = qt.TensorNetwork(all_tensors)

    print(f"  env_tn tensors: {[t.inds for t in env_tn]}")
    print(f"  env_tn outer_inds: {list(env_tn.outer_inds())}")

    # Contract - now 'r' should be outer since each A_j has its own 'r'
    # Wait, no - they all share 'r'!
    # The issue is that A_1[s, r] and A_2[s, r, out] both have 'r'
    # When we combine them in a TensorNetwork, 'r' becomes internal

    # We need to contract while keeping 'r' open
    # Request output_inds explicitly
    target_inds = list(tn[f"Node{target_idx}"].inds)
    output_inds = [batch_dim] + target_inds

    # But we also need 'out' if target doesn't have it
    if "out" not in target_inds:
        output_inds.append("out")

    print(f"  Requesting output_inds: {output_inds}")

    env = env_tn.contract(output_inds=output_inds)
    print(f"  Result: shape={env.shape}, inds={env.inds}")

    return env


print("\nComputing environment for Node0:")
env0 = cpda_environment(tn, inputs, 0, L)

print("\nComputing environment for Node1:")
env1 = cpda_environment(tn, inputs, 1, L)

print("\nComputing environment for Node2 (output node):")
env2 = cpda_environment(tn, inputs, 2, L)
