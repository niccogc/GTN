#type: ignore
import torch
from model.standard.SymMPO2 import SymMPO2
from model.base.SymNTN import SymNTN
from model.losses import MSELoss
from model.utils import create_inputs
torch.manual_seed(42)
L = 5
phys_dim = 5
model = SymMPO2(L=L, bond_dim=4, phys_dim=phys_dim, output_dim=1)
loader = create_inputs(
    X=torch.randn(16, 4), y=torch.randn(16, 1),
    input_labels=model.input_labels,
    output_labels=model.output_dims,
    batch_size=16,
    append_bias=True,
)
trainer = SymNTN(
    model=model,
    output_dims=model.output_dims,
    input_dims=model.input_dims,
    loss=MSELoss(),
    data_stream=loader,
)

print(trainer.forward(trainer.tn, trainer.data.data_mu))
# print('Testing A0 (tied parameter):')
# J_A, H_A = trainer._compute_H_b('Node0')
# print(f'  J_A shape: {J_A.shape}, inds: {J_A.inds}')
# print(f'  H_A shape: {H_A.shape}, inds: {H_A.inds}')
# print('\\nTesting B (single node):')
# J_B, H_B = trainer._compute_H_b('Node1')
# print(f'  J_B shape: {J_B.shape}, inds: {J_B.inds}')
# print(f'  H_B shape: {H_B.shape}, inds: {H_B.inds}')
