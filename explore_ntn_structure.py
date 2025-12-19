"""
Exploration script to understand NTN's batch_forward and batch_environment methods.
This helps us understand what indices, shapes, and structures are involved.
"""
import torch
import quimb.tensor as qt
from model.NTN import NTN
from model.builder import Inputs
from model.losses import MSELoss

torch.set_default_dtype(torch.float64)

print("="*80)
print("EXPLORING NTN BATCH_FORWARD AND BATCH_ENVIRONMENT")
print("="*80)

# === 1. Create a simple MPS-like structure ===
print("\n" + "="*80)
print("1. CREATING SIMPLE MPS STRUCTURE")
print("="*80)

N_SAMPLES = 100
BATCH_SIZE = 50
D_bond = 3
D_phys = 2

# Generate simple data
x_raw = torch.randn(N_SAMPLES, 1)
y_raw = x_raw**2
x_features = torch.cat([x_raw, torch.ones_like(x_raw)], dim=1)

print(f"Data shapes: X={x_features.shape}, Y={y_raw.shape}")

# Create MPS structure - following the pattern from test_simple_comparison.py
def init_weights(shape):
    w = torch.randn(*shape) * 0.1
    return w / torch.norm(w)

# Physical indices: x1, x2, x3
# Bond indices: b1, b2
# Output index: y
input_labels = ["x1", "x2", "x3"]

t1 = qt.Tensor(data=init_weights((D_phys, D_bond)), inds=('x1', 'b1'), tags={'Node1'})
t2 = qt.Tensor(data=init_weights((D_bond, D_phys, D_bond, 1)), inds=('b1', 'x2', 'b2', 'y'), tags={'Node2'})
t3 = qt.Tensor(data=init_weights((D_bond, D_phys)), inds=('b2', 'x3'), tags={'Node3'})

tn = qt.TensorNetwork([t1, t2, t3])

print("\nTensor Network Structure:")
for i, tensor in enumerate(tn):
    print(f"  Tensor {i}: inds={tensor.inds}, shape={tensor.shape}, tags={tensor.tags}")

print(f"\nOuter indices (connect to inputs/outputs): {tn.outer_inds()}")
print(f"Inner indices (bond dimensions): {tn.inner_inds()}")

# === 2. Setup NTN ===
print("\n" + "="*80)
print("2. SETTING UP NTN")
print("="*80)

loader = Inputs(
    inputs=[x_features],
    outputs=[y_raw],
    outputs_labels=["y"],
    input_labels=input_labels,
    batch_dim="batch",
    batch_size=BATCH_SIZE
)

model = NTN(
    tn=tn,
    output_dims=["y"],
    input_dims=input_labels,
    loss=MSELoss(),
    data_stream=loader,
    method='cholesky'
)

print(f"\nNTN configuration:")
print(f"  input_dims: {model.input_dims}")
print(f"  output_dims: {model.output_dims}")
print(f"  batch_dim: {model.batch_dim}")

# === 3. Explore batch_forward ===
print("\n" + "="*80)
print("3. EXPLORING BATCH_FORWARD")
print("="*80)

# Get a single batch - following the pattern from test
batch_data = next(iter(loader))
inputs_list, y_batch = batch_data
inputs_batch = inputs_list[0]  # First (and only) input array

print(f"\nBatch data:")
print(f"  inputs_batch shape: {inputs_batch.shape}")
print(f"  y_batch shape: {y_batch.shape}")

# Create input tensors - FOLLOWING THE CORRECT PATTERN
print(f"\nCreating input tensors with batch_dim='{model.batch_dim}':")
input_tensors = [
    qt.Tensor(inputs_batch, inds=["batch", f"{i}"], tags=f"Input_{i}") 
    for i in input_labels
]

for idx, inp in enumerate(input_tensors):
    print(f"  Input tensor {idx}: inds={inp.inds}, shape={inp.shape}, tags={inp.tags}")

# Call batch_forward
print(f"\nCalling _batch_forward with output_inds={[model.batch_dim] + model.output_dimensions}:")
result = model._batch_forward(
    input_tensors, 
    model.tn, 
    output_inds=[model.batch_dim] + model.output_dimensions
)

print(f"  Result: inds={result.inds}, shape={result.shape}")
print(f"  Result data sample: {result.data[:3]}")

# === 4. Explore batch_environment ===
print("\n" + "="*80)
print("4. EXPLORING BATCH_ENVIRONMENT")
print("="*80)

target_node = 'Node2'
print(f"\nComputing environment for target_node='{target_node}'")

env = model._batch_environment(
    input_tensors,
    model.tn,
    target_tag=target_node,
    sum_over_batch=False,
    sum_over_output=False
)

print(f"  Environment tensor:")
print(f"    inds: {env.inds}")
print(f"    shape: {env.shape}")
print(f"    tags: {env.tags}")

# Get the target node to see what indices it connects to
target_tensor = model.tn[target_node]
print(f"\n  Target node '{target_node}':")
print(f"    inds: {target_tensor.inds}")
print(f"    shape: {target_tensor.shape}")

# The environment should have indices that connect to the target node
print(f"\n  Environment connection to target:")
common_inds = set(env.inds) & set(target_tensor.inds)
print(f"    Common (bond) indices: {common_inds}")
print(f"    These bond indices allow env to contract with target node")

# === 5. Test with sum_over_batch and sum_over_output ===
print("\n" + "="*80)
print("5. TESTING DIFFERENT ENVIRONMENT MODES")
print("="*80)

configs = [
    (False, False, "Full environment (batch & output)"),
    (True, False, "Sum over batch, keep output"),
    (False, True, "Keep batch, sum over output"),
    (True, True, "Sum over batch & output"),
]

for sum_batch, sum_output, desc in configs:
    env = model._batch_environment(
        input_tensors,
        model.tn,
        target_tag=target_node,
        sum_over_batch=sum_batch,
        sum_over_output=sum_output
    )
    print(f"\n  {desc}:")
    print(f"    sum_over_batch={sum_batch}, sum_over_output={sum_output}")
    print(f"    inds: {env.inds}")
    print(f"    shape: {env.shape}")
    
    # Show what indices are present
    has_batch = model.batch_dim in env.inds
    has_output = any(out in env.inds for out in model.output_dimensions)
    bond_inds = set(env.inds) & set(target_tensor.inds)
    print(f"    has batch_dim: {has_batch}, has output: {has_output}")
    print(f"    bond indices to target: {bond_inds}")

# === 6. Understand full derivative computation ===
print("\n" + "="*80)
print("6. UNDERSTANDING DERIVATIVE COMPUTATION")
print("="*80)

print(f"\nFor node '{target_node}', the derivative computation involves:")
print(f"  1. Compute environment E (excluding target node)")
print(f"     E.inds = {env.inds}")
print(f"  2. Reconstruct y_pred = contract(E, target_node)")
print(f"  3. Compute loss derivatives dL/dy, d²L/dy²")
print(f"  4. Compute node gradient: contract(E, dL/dy)")
print(f"  5. Compute node Hessian: contract(E, d²L/dy², E')")

# Test environment contraction with target
print(f"\n  Testing: Can environment contract with target node?")
test_tn = env & target_tensor
print(f"    Combined TN outer inds: {test_tn.outer_inds()}")
print(f"    Should contain batch and output: {model.batch_dim in test_tn.outer_inds() and 'y' in test_tn.outer_inds()}")

# === 7. Show how inputs work for different nodes ===
print("\n" + "="*80)
print("7. UNDERSTANDING INPUT STRUCTURE")
print("="*80)

print(f"\nInput data shape: {inputs_batch.shape} -> (batch_size, num_features)")
print(f"num_features = {inputs_batch.shape[1]} (one column per input label)")
print(f"\nEach input tensor:")
for i, (label, inp_tensor) in enumerate(zip(input_labels, input_tensors)):
    print(f"  '{label}': extracts column {i} from inputs_batch")
    print(f"    inds={inp_tensor.inds}, shape={inp_tensor.shape}")
    print(f"    data[:3, {i}] = {inputs_batch[:3, i]}")

# === 8. Summary ===
print("\n" + "="*80)
print("8. KEY FINDINGS SUMMARY")
print("="*80)

print("""
STRUCTURE:
- Physical indices (x1, x2, x3): Connect to input data columns
- Bond indices (b1, b2): Connect MPS nodes together (internal)
- Batch dimension: Runs through all data samples
- Output indices (y): The prediction target

INPUT TENSORS:
- Created from same data array (inputs_batch)
- Each gets different physical index (x1, x2, x3)
- All share batch dimension
- Shape: (batch_size, phys_dim) where phys_dim matches input data columns

BATCH_FORWARD:
- Contracts inputs with TN nodes via physical indices
- Contracts bonds between nodes
- Returns prediction with shape (batch_size, output_dim)

BATCH_ENVIRONMENT:
- Removes target node, contracts everything else
- Keeps bond indices that connect to target
- Can sum/keep batch and output dimensions as needed

MPS OPTIMIZATION:
- Sequential contraction (left-to-right) efficient for chains
- Can cache left/right environments (DMRG-style)
- Canonical forms improve numerical stability
""")

print("\n" + "="*80)
print("EXPLORATION COMPLETE - See NTN_MPS_GUIDE.md for full documentation")
print("="*80)
