import torch
from model.standard.SymMPO2 import SymMPO2
from model.base.SymNTN import SymNTN
from model.losses import MSELoss
from model.utils import create_inputs
torch.manual_seed(42)
L = 3
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
# Debug one batch
batch = next(iter(loader.data_mu_y))
inputs, y_true = batch
canonical_tag = 'A0'
positions = [0, 2]
ref_pos = 0
# Get env for pos=0
node_tag = 'Node0'
env = trainer._batch_environment(
    inputs, trainer.tn, target_tag=node_tag,
    sum_over_batch=False, sum_over_output=False
)
print(f'env for Node0: inds={env.inds}')
env_canonical = trainer._transform_env_to_canonical(env, 0, canonical_tag, prime=False)
env_prime = trainer._transform_env_to_canonical(env, 0, canonical_tag, prime=True)
print(f'env_canonical: inds={env_canonical.inds}')
print(f'env_prime: inds={env_prime.inds}')
# d2L_tensor
canonical_inds = trainer.tn['Node0'].inds
print(f'canonical_inds: {canonical_inds}')
out_inds = trainer.output_dimensions
out_col_inds = [x + '_prime' for x in out_inds]
print(f'out_inds: {out_inds}, out_col_inds: {out_col_inds}')
hess_out_inds = list(canonical_inds) + [f'{x}_prime' for x in canonical_inds]
print(f'hess_out_inds: {hess_out_inds}')
