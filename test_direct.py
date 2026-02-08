# type: ignore
import torch
import sys

torch.set_default_dtype(torch.float64)

from model.base.NTN import NTN
from model.losses import MSELoss
from model.utils import create_inputs
from model.standard import MPO2
from experiments.dataset_loader import load_dataset

print("Loading concrete dataset...")
data, info = load_dataset("concrete")
print(f"Train: {data['X_train'].shape}, Val: {data['X_val'].shape}")

L = 3
bond_dim = 4
input_dim = data["X_train"].shape[1]

model = MPO2(L=L, bond_dim=bond_dim, phys_dim=input_dim, output_dim=1)
print(f"Model: L={L}, bond_dim={bond_dim}, phys_dim={input_dim}")

loader = create_inputs(
    X=data["X_train"][:50],
    y=data["y_train"][:50],
    input_labels=model.input_labels,
    output_labels=model.output_dims,
    batch_size=16,
    append_bias=False,
)
print(f"Batches: {len(loader)}")

loss_fn = MSELoss()
ntn = NTN(
    tn=model.tn,
    output_dims=model.output_dims,
    input_dims=model.input_dims,
    loss=loss_fn,
    data_stream=loader,
)

print("\n" + "=" * 60)
print("Calling _compute_H_b for Node2...")
print("=" * 60)
sys.stdout.flush()

b, H = ntn._compute_H_b("Node2")

print("\n" + "=" * 60)
print("RESULTS:")
print(f"b.inds: {b.inds}, b.shape: {b.shape}")
print(f"H.inds: {H.inds}, H.shape: {H.shape}")
print("=" * 60)
