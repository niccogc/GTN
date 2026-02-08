# type: ignore
import torch
import numpy as np

torch.set_default_dtype(torch.float64)

from model.NTN import NTN
from model.losses import MSELoss, CrossEntropyLoss
from model.utils import create_inputs
from model.standard import MPO2

np.random.seed(42)
torch.manual_seed(42)

n_samples = 20
n_classes = 3
batch_size = 8

L = 3
bond_dim = 4
phys_dim = 5

X = torch.randn(n_samples, L * phys_dim)
y = torch.randint(0, n_classes, (n_samples,))
y_onehot = torch.zeros(n_samples, n_classes)
y_onehot.scatter_(1, y.unsqueeze(1), 1)

print("=" * 60)
print("TEST SETUP")
print("=" * 60)
print(f"n_samples: {n_samples}")
print(f"L (sites): {L}, phys_dim: {phys_dim}")
print(f"X.shape: {X.shape} (n_samples, L*phys_dim)")
print(f"n_classes: {n_classes}")
print(f"batch_size: {batch_size}")
print(f"Expected batches: {(n_samples + batch_size - 1) // batch_size}")
print()

model = MPO2(
    L=L,
    bond_dim=bond_dim,
    phys_dim=phys_dim,
    output_dim=n_classes,
    output_site=L - 1,
)

print("Model created:")
print(f"  L={L}, bond_dim={bond_dim}, output_site={L - 1}")
print(f"  input_labels: {model.input_labels}")
print(f"  output_dims: {model.output_dims}")
print()

# Create data loader
loader = create_inputs(
    X=X,
    y=y_onehot,
    input_labels=model.input_labels,
    output_labels=model.output_dims,
    batch_size=batch_size,
)

# Create NTN with CrossEntropy (classification)
loss_fn = CrossEntropyLoss()
ntn = NTN(
    tn=model.tn,
    output_dims=model.output_dims,
    input_dims=model.input_dims,
    loss=loss_fn,
    data_stream=loader,
)

print("=" * 60)
print("TESTING _compute_H_b (calls _batch_node_derivatives)")
print("=" * 60)

# Test on the output node (Node2)
node_tag = "Node2"
print(f"\nComputing H and b for node: {node_tag}")
print("-" * 60)

b, H = ntn._compute_H_b(node_tag)

print("\n" + "=" * 60)
print("FINAL RESULTS")
print("=" * 60)
print(f"b (gradient): inds={b.inds}, shape={b.shape}")
print(f"H (hessian):  inds={H.inds}, shape={H.shape}")
print()
print("Check: batch_dim 's' should NOT be in final indices!")
print(f"  's' in b.inds: {'s' in b.inds}")
print(f"  's' in H.inds: {'s' in H.inds}")
