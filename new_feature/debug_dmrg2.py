import sys
sys.path.insert(0, '/home/nicci/Desktop/remote/GTN')

import torch
import numpy as np

from model.standard.TNML import TNML_P
from model.base.DMRG import DMRG
from model.base.NTN import NTN
from model.utils import create_inputs
from model.losses import MSELoss

torch.manual_seed(42)
np.random.seed(42)

n_features = 4
n_samples = 100
output_dim = 1

X = torch.randn(n_samples, n_features)
y = X.sum(dim=1, keepdim=True) + 0.1 * torch.randn(n_samples, 1)

X_train, X_val = X[:80], X[80:]
y_train, y_val = y[:80], y[80:]

model = TNML_P(
    L=3,
    bond_dim=4,
    phys_dim=n_features,
    output_dim=output_dim,
    output_site=n_features - 1,
)

print("Model structure:")
for tensor in model.tn:
    print(f"  {list(tensor.tags)[0]}: {tensor.inds} {tensor.shape}")

loader_train = create_inputs(
    X=X_train, y=y_train,
    input_labels=model.input_labels,
    output_labels=model.output_dims,
    batch_size=32,
    append_bias=False,
    encoding="polynomial",
    poly_degree=model.poly_degree,
)

loader_val = create_inputs(
    X=X_val, y=y_val,
    input_labels=model.input_labels,
    output_labels=model.output_dims,
    batch_size=32,
    append_bias=False,
    encoding="polynomial",
    poly_degree=model.poly_degree,
)

print("\n" + "="*60)
print("Testing with NTN (1-site, known working)")
print("="*60)

model_ntn = TNML_P(
    L=3, bond_dim=4, phys_dim=n_features, output_dim=output_dim,
    output_site=n_features - 1,
)

loader_train_ntn = create_inputs(
    X=X_train, y=y_train,
    input_labels=model_ntn.input_labels,
    output_labels=model_ntn.output_dims,
    batch_size=32, append_bias=False,
    encoding="polynomial", poly_degree=model_ntn.poly_degree,
)

loader_val_ntn = create_inputs(
    X=X_val, y=y_val,
    input_labels=model_ntn.input_labels,
    output_labels=model_ntn.output_dims,
    batch_size=32, append_bias=False,
    encoding="polynomial", poly_degree=model_ntn.poly_degree,
)

ntn = NTN(
    tn=model_ntn.tn,
    output_dims=model_ntn.output_dims,
    input_dims=model_ntn.input_dims,
    loss=MSELoss(),
    data_stream=loader_train_ntn,
)

scores_train, scores_val = ntn.fit(
    n_epochs=5,
    regularize=True,
    jitter=1e-4,
    verbose=True,
    val_data=loader_val_ntn,
)

print(f"\nNTN Final: train_loss={scores_train['loss']:.4f}, val_loss={scores_val['loss']:.4f}")

print("\n" + "="*60)
print("Testing with DMRG (2-site)")
print("="*60)

model_dmrg = TNML_P(
    L=3, bond_dim=4, phys_dim=n_features, output_dim=output_dim,
    output_site=n_features - 1,
)

loader_train_dmrg = create_inputs(
    X=X_train, y=y_train,
    input_labels=model_dmrg.input_labels,
    output_labels=model_dmrg.output_dims,
    batch_size=32, append_bias=False,
    encoding="polynomial", poly_degree=model_dmrg.poly_degree,
)

loader_val_dmrg = create_inputs(
    X=X_val, y=y_val,
    input_labels=model_dmrg.input_labels,
    output_labels=model_dmrg.output_dims,
    batch_size=32, append_bias=False,
    encoding="polynomial", poly_degree=model_dmrg.poly_degree,
)

dmrg = DMRG(
    tn=model_dmrg.tn,
    output_dims=model_dmrg.output_dims,
    input_dims=model_dmrg.input_dims,
    loss=MSELoss(),
    data_stream=loader_train_dmrg,
)

scores_train, scores_val = dmrg.fit(
    n_epochs=5,
    regularize=True,
    jitter=1e-4,
    verbose=True,
    val_data=loader_val_dmrg,
    max_bond=8,
    cutoff=1e-10,
)

print(f"\nDMRG Final: train_loss={scores_train['loss']:.4f}, val_loss={scores_val['loss']:.4f}")
