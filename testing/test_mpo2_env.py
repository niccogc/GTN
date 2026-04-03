# type: ignore
"""
Check how MPO2 environment works for the output node.
"""

import torch
import quimb.tensor as qt
from model.standard import MPO2

# Create MPO2
L = 4
bond_dim = 3
phys_dim = 5
output_dim = 2
batch_size = 8

print("=" * 60)
print("MPO2 STRUCTURE")
print("=" * 60)

mpo2 = MPO2(L=L, bond_dim=bond_dim, phys_dim=phys_dim, output_dim=output_dim)
tn = mpo2.tn

print(f"Input labels: {mpo2.input_labels}")
print(f"Input dims: {mpo2.input_dims}")
print(f"Output dims: {mpo2.output_dims}")
print(f"Output site: {mpo2.output_site}")
print()

print("Tensors:")
for tensor in tn:
    print(f"  {list(tensor.tags)[0]}: shape={tensor.shape}, inds={tensor.inds}")

print("\n" + "=" * 60)
print("CREATE INPUTS")
print("=" * 60)

inputs = []
for i in range(L):
    data = torch.randn(batch_size, phys_dim)
    tensor = qt.Tensor(data=data, inds=("s", f"x{i}"), tags={f"Input{i}"})
    inputs.append(tensor)
    print(f"Input{i}: shape={tensor.shape}, inds={tensor.inds}")

batch_dim = "s"

print("\n" + "=" * 60)
print("ENVIRONMENT FOR OUTPUT NODE (Node3)")
print("=" * 60)

target_tag = f"Node{mpo2.output_site}"
target_tensor = tn[target_tag]
print(f"Target: {target_tag}")
print(f"  Shape: {target_tensor.shape}")
print(f"  Inds: {target_tensor.inds}")

# Create env_tn
env_tn = tn.copy() & inputs
env_tn.delete(target_tag)

print(f"\nenv_tn outer_inds: {list(env_tn.outer_inds())}")

# Contract with default outer_inds
env_default = env_tn.contract(output_inds=list(env_tn.outer_inds()))
print(f"Env (default): shape={env_default.shape}, inds={env_default.inds}")

# What indices does output node have?
# For MPO2 output node: (b{i-1}, x{i}, out) - so env should have (s, b{i-1}, x{i})
# because 'out' comes from the node itself

print("\n" + "=" * 60)
print("ENVIRONMENT FOR NON-OUTPUT NODE (Node0)")
print("=" * 60)

target_tag = "Node0"
target_tensor = tn[target_tag]
print(f"Target: {target_tag}")
print(f"  Shape: {target_tensor.shape}")
print(f"  Inds: {target_tensor.inds}")

env_tn = tn.copy() & inputs
env_tn.delete(target_tag)

print(f"\nenv_tn outer_inds: {list(env_tn.outer_inds())}")

env_default = env_tn.contract(output_inds=list(env_tn.outer_inds()))
print(f"Env (default): shape={env_default.shape}, inds={env_default.inds}")

# For MPO2 Node0: (x0, b0) - so env should have (s, x0, b0, out)
# because 'out' is in another node in the environment

print("\n" + "=" * 60)
print("KEY OBSERVATION")
print("=" * 60)
print("""
For MPO2:
- Output node has indices (bond, phys, out)
- Environment for output node has outer_inds that include bond and phys
- The 'out' index is NOT in outer_inds because it's in the target node

For CPDA:
- Output node has indices (phys, r, out) 
- Non-output nodes have indices (phys, r)
- The 'r' index is SHARED between all nodes
- When we remove output node, 'r' is still shared between remaining nodes
- So 'r' is NOT in outer_inds (it's internal)

This is the difference! In MPO2, bond indices are only shared between 2 adjacent nodes.
In CPDA, 'r' is shared between ALL nodes.
""")
