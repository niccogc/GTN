# type: ignore
"""
Test that updated NTN._batch_environment works for both MPO2 and CPDA.
"""

import torch
import quimb.tensor as qt
from model.standard import MPO2
from model.base.NTN import NTN
from model.losses import MSELoss
from model.utils import create_inputs

# Parameters
L = 4
bond_dim = 3
input_features = 5  # Original feature count (before bias)
phys_dim = input_features + 1  # +1 for bias term
output_dim = 1  # Regression
batch_size = 16
n_samples = 64

torch.manual_seed(42)

print("=" * 70)
print("TEST 1: MPO2 with updated NTN")
print("=" * 70)

# Create MPO2 with phys_dim including bias
mpo2 = MPO2(L=L, bond_dim=bond_dim, phys_dim=phys_dim, output_dim=output_dim)

# Create synthetic data (before bias)
X = torch.randn(n_samples, input_features)
y = torch.randn(n_samples, output_dim)

# Create inputs
loader = create_inputs(
    X=X,
    y=y,
    input_labels=mpo2.input_labels,
    output_labels=mpo2.output_dims,
    batch_size=batch_size,
    append_bias=True,
)

# Create NTN
ntn = NTN(
    tn=mpo2.tn,
    output_dims=mpo2.output_dims,
    input_dims=mpo2.input_dims,
    loss=MSELoss(),
    data_stream=loader,
)

# Test environment computation for each node
print("\nTesting environment computation:")
for i in range(L):
    target_tag = f"Node{i}"

    # Get first batch
    batch_data = next(iter(loader.data_mu))
    inputs = batch_data

    env = ntn._batch_environment(
        inputs, ntn.tn, target_tag, sum_over_batch=False, sum_over_output=False
    )

    # Verify forward from env
    target_tensor = ntn.tn[target_tag]
    forward_tn = env & target_tensor
    result = forward_tn.contract(output_inds=[ntn.batch_dim] + ntn.output_dimensions)

    print(f"  {target_tag}: env shape={env.shape}, forward shape={result.shape}")

# Test a few training steps
print("\nTesting training (2 epochs):")
try:
    scores_train, scores_val = ntn.fit(n_epochs=2, verbose=False, jitter=1e-4)
    print(f"  Final train loss: {scores_train['loss']:.6f}")
    print("  MPO2 training: PASSED")
except Exception as e:
    print(f"  MPO2 training FAILED: {e}")

print("\n" + "=" * 70)
print("TEST 2: CPDA with updated NTN")
print("=" * 70)

# Create CPDA TN manually
rank = bond_dim
output_site = L - 1

tensors_cpda = []
for i in range(L):
    if i == output_site:
        shape = (phys_dim, rank, output_dim)  # phys_dim already includes bias
        inds = (f"x{i}", "r", "out")
    else:
        shape = (phys_dim, rank)
        inds = (f"x{i}", "r")

    data = torch.randn(*shape) * 0.1
    tensor = qt.Tensor(data=data, inds=inds, tags={f"Node{i}"})
    tensors_cpda.append(tensor)

tn_cpda = qt.TensorNetwork(tensors_cpda)

# CPDA model attributes
cpda_input_labels = [f"x{i}" for i in range(L)]
cpda_input_dims = [f"x{i}" for i in range(L)]
cpda_output_dims = ["out"]

# Create inputs for CPDA
loader_cpda = create_inputs(
    X=X,
    y=y,
    input_labels=cpda_input_labels,
    output_labels=cpda_output_dims,
    batch_size=batch_size,
    append_bias=True,
)

# Create NTN for CPDA
ntn_cpda = NTN(
    tn=tn_cpda,
    output_dims=cpda_output_dims,
    input_dims=cpda_input_dims,
    loss=MSELoss(),
    data_stream=loader_cpda,
)

# Test environment computation
print("\nTesting environment computation:")
for i in range(L):
    target_tag = f"Node{i}"

    batch_data = next(iter(loader_cpda.data_mu))
    inputs = batch_data

    env = ntn_cpda._batch_environment(
        inputs, ntn_cpda.tn, target_tag, sum_over_batch=False, sum_over_output=False
    )

    target_tensor = ntn_cpda.tn[target_tag]
    forward_tn = env & target_tensor
    result = forward_tn.contract(output_inds=[ntn_cpda.batch_dim] + ntn_cpda.output_dimensions)

    is_output = i == output_site
    print(
        f"  {target_tag} (output={is_output}): env shape={env.shape}, forward shape={result.shape}"
    )

# Test training
print("\nTesting training (2 epochs):")
try:
    scores_train, scores_val = ntn_cpda.fit(n_epochs=2, verbose=False, jitter=1e-4)
    print(f"  Final train loss: {scores_train['loss']:.6f}")
    print("  CPDA training: PASSED")
except Exception as e:
    print(f"  CPDA training FAILED: {e}")
    import traceback

    traceback.print_exc()

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
