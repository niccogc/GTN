# type: ignore
"""
Debug script to understand CPDA contraction behavior step by step.
"""

import torch
import quimb.tensor as qt

# Simple CPDA: 3 nodes, rank=2, phys_dim=3, output_dim=2
L = 3
rank = 2
phys_dim = 3
output_dim = 2
batch_size = 4

print("=" * 60)
print("CREATING CPDA TENSOR NETWORK")
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
print(f"\nTN outer_inds: {list(tn.outer_inds())}")

print("\n" + "=" * 60)
print("CREATING INPUT TENSORS")
print("=" * 60)

inputs = []
for i in range(L):
    data = torch.randn(batch_size, phys_dim)
    tensor = qt.Tensor(data=data, inds=("s", f"x{i}"), tags={f"Input{i}"})
    inputs.append(tensor)
    print(f"Input{i}: shape={tensor.shape}, inds={tensor.inds}")

print("\n" + "=" * 60)
print("TEST: FULL FORWARD PASS")
print("=" * 60)

full_tn = tn & inputs
print(f"Full TN outer_inds: {list(full_tn.outer_inds())}")

# Contract with output_inds to keep s and out
result = full_tn.contract(output_inds=["s", "out"])
print(f"Result: shape={result.shape}, inds={result.inds}")

print("\n" + "=" * 60)
print("TEST: ENVIRONMENT FOR Node0 (non-output node)")
print("=" * 60)

target_tag = "Node0"
target_tensor = tn[target_tag]
print(f"Target: {target_tag}, shape={target_tensor.shape}, inds={target_tensor.inds}")

# Create env_tn = tn + inputs, then remove target
env_tn = tn.copy() & inputs
print(f"\nBefore delete - env_tn tensors:")
for t in env_tn:
    print(f"  {list(t.tags)}: shape={t.shape}, inds={t.inds}")

env_tn.delete(target_tag)
print(f"\nAfter delete - env_tn tensors:")
for t in env_tn:
    print(f"  {list(t.tags)}: shape={t.shape}, inds={t.inds}")

print(f"\nenv_tn outer_inds: {list(env_tn.outer_inds())}")

# Now contract - what output_inds should we use?
# We want to keep: s (batch), x0 (target's physical), r (target's rank)
# But wait - r is SHARED between Node1 and Node2, so it's NOT an outer index!

print("\n--- Contracting with default (all outer) ---")
outer = list(env_tn.outer_inds())
env_default = env_tn.contract(output_inds=outer)
print(f"Default contraction: shape={env_default.shape}, inds={env_default.inds}")

print("\n--- What indices does the target node have? ---")
print(f"Target inds: {target_tensor.inds}")  # ('x0', 'r')

print("\n--- The problem: 'r' is shared between remaining nodes! ---")
print("When we remove Node0, Node1 and Node2 still share 'r'")
print("So 'r' gets contracted when we contract env_tn")

print("\n" + "=" * 60)
print("UNDERSTANDING THE CPDA STRUCTURE ISSUE")
print("=" * 60)

print("""
In MPS/MPO, bond indices connect ADJACENT nodes:
  Node0 --b0-- Node1 --b1-- Node2
  
When we remove Node1, the environment has open indices b0 and b1.

In CPDA, ALL nodes share the SAME rank index 'r':
  Node0 --r
  Node1 --r  (all connected to same 'r')
  Node2 --r
  
When we remove Node0, Node1 and Node2 STILL share 'r'.
So when we contract them, 'r' gets summed over!

This means the standard NTN environment computation WON'T work for CPDA!
""")

print("\n" + "=" * 60)
print("WHAT CPDA ENVIRONMENT SHOULD LOOK LIKE")
print("=" * 60)

print("""
For CPDA, the "environment" of Node_i should be:
  env_i = Input0 @ Node0 * Input1 @ Node1 * ... (excluding i) * Inputs @ Nodes
  
But we DON'T want to contract over 'r' - we want elementwise product over r!

Environment shape for Node0 should be: (batch, x0, r, out)
  - batch 's' from inputs
  - x0 from target node (to contract with target)
  - r from target node (to contract with target)  
  - out from output node

Let's manually compute this:
""")

# Manual CPDA environment for Node0
# Step 1: Contract each non-target node with its input
print("Step 1: Contract each node with its input (keeping r open)")
contracted = []
for i in range(L):
    if i == 0:  # skip target
        continue
    node = tn[f"Node{i}"]
    inp = inputs[i]
    # Contract over x{i}, keeping s and r (and out if present)
    node_with_input = node & inp
    result_i = node_with_input.contract()
    print(f"  Node{i} @ Input{i}: shape={result_i.shape}, inds={result_i.inds}")
    contracted.append(result_i)

print("\nStep 2: Combine contracted tensors")
# Now we have tensors with indices (s, r) and (s, r, out)
# We want elementwise product over r, sum nothing
combined = contracted[0]
for c in contracted[1:]:
    combined = combined & c

print(f"Combined before contract: outer_inds={list(combined.outer_inds())}")

# Contract - but what should remain open?
# We want (s, r, out) for output node's environment, or (s, r) for others
# BUT 's' appears multiple times! Each Input has 's'.

print("\n--- The 's' index appears in multiple input tensors! ---")
print("This is the key insight: all inputs share the batch index 's'")
print("When we contract, 's' should remain as a single outer index")

# The environment for Node0 needs indices (s, x0, r, out) to contract with Node0
# But x0 is ONLY in Input0, not in any remaining node!
# So we need Input0 as well in the environment

print("\n" + "=" * 60)
print("CORRECT CPDA ENVIRONMENT COMPUTATION")
print("=" * 60)

print("""
The environment for Node_i must include Input_i!
Because Input_i provides the x_i index that will contract with Node_i.

env_i = Input_i * (product over j != i of: Input_j @ Node_j)

Let's verify:
""")

# Correct environment for Node0
target_input = inputs[0]  # Input0 with indices (s, x0)
print(f"Target input: shape={target_input.shape}, inds={target_input.inds}")

# Build rest: for each j != 0, contract Input_j with Node_j
rest_tensors = [target_input]  # Start with target's input
for i in range(L):
    if i == 0:
        continue
    rest_tensors.append(tn[f"Node{i}"])
    rest_tensors.append(inputs[i])

env_tn_correct = qt.TensorNetwork(rest_tensors)
print(f"\nCorrect env_tn outer_inds: {list(env_tn_correct.outer_inds())}")

# Contract keeping s, x0, r, out
# Wait - 'r' is still shared between Node1 and Node2!
# So it will still be contracted...

print("\n--- Still have the 'r' problem! ---")

# Let me check what the actual outer inds are
print(f"Outer inds: {list(env_tn_correct.outer_inds())}")
# This should show 'x0' and 'out', but 'r' is internal (shared between Node1, Node2)

# Contract
env_correct = env_tn_correct.contract(output_inds=list(env_tn_correct.outer_inds()))
print(f"Contracted: shape={env_correct.shape}, inds={env_correct.inds}")

print("\n" + "=" * 60)
print("CONCLUSION")
print("=" * 60)
print("""
The CPDA structure has a fundamental difference from MPS:
- In MPS, removing a node leaves OPEN bond indices
- In CPDA, removing a node still leaves 'r' as SHARED (internal) between other nodes

This means:
1. Standard NTN environment computation contracts over 'r' 
2. The environment doesn't have the right shape to contract with the target node
3. We NEED a custom CPDA_NTN class that handles this differently!

The CPDA forward pass works (r gets contracted with output_inds=['s', 'out'])
But the environment computation needs special handling.
""")
