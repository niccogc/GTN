# type: ignore
import quimb.tensor as qb
import torch
from batch_moving_environment import BatchMovingEnvironment 

# 1. Setup
qb.set_tensor_linop_backend('torch')

L = 3
BOND_DIM = 4
DIM_PIXELS = 3
DIM_PATCHES = 3
N_OUTPUTS = 5
BATCH = 10

# MPS & Inputs
psi = qb.MPS_rand_state(L, bond_dim=BOND_DIM, phys_dim=DIM_PIXELS)
phi = qb.MPS_rand_state(L, bond_dim=BOND_DIM, phys_dim=DIM_PATCHES)
psi[f"I{1}"].new_ind('out', size=N_OUTPUTS, axis=-1, mode='random')

psi.apply_to_arrays(lambda x: torch.tensor(x, dtype=torch.float32))
phi.apply_to_arrays(lambda x: torch.tensor(x, dtype=torch.float32))

psi.reindex({f"k{i}": f"{i}_pixels" for i in range(L)}, inplace=True)
phi.reindex({f"k{i}": f"{i}_patches" for i in range(L)}, inplace=True)

inputs = [
    qb.Tensor(torch.randn(BATCH, DIM_PIXELS, DIM_PATCHES), inds=['s', f'{i}_pixels', f'{i}_patches'], tags={f'I{i}', 'OP'}) 
    for i in range(L)
]

tn = psi & phi
for inp in inputs:
    tn.add_tensor(inp, virtual=True)

# 2. Init
print("--- FULL DEBUG INSPECTION ---")
env = BatchMovingEnvironment(tn, begin='left', bsz=1, batch_inds=['s'], output_dims=['out'])

def dump_env(pos):
    """Prints every tensor in the environment at 'pos'."""
    print(f"\n[POS {pos}] DUMPING ALL TENSORS IN ENV:")
    current_env = env.envs[pos]
    
    # 1. Check Tags & Duplicates
    lefts = [t for t in current_env if "_LEFT" in t.tags]
    rights = [t for t in current_env if "_RIGHT" in t.tags]
    
    if len(lefts) > 1: print(f"  !!! CRITICAL ERROR: Found {len(lefts)} tensors tagged '_LEFT' (Should be 0 or 1)")
    if len(rights) > 1: print(f"  !!! CRITICAL ERROR: Found {len(rights)} tensors tagged '_RIGHT' (Should be 0 or 1)")

    # 2. Print Details
    for i, t in enumerate(current_env.tensors):
        # We simplify tags for readability
        tags = list(t.tags)
        print(f"  T{i}: Tags={tags} | Shape={t.shape} | Inds={t.inds}")

# --- TEST SEQUENCE ---

dump_env(0)

print("\n>>> MOVING RIGHT TO 1...")
env.move_right()
dump_env(1)

print("\n>>> MOVING RIGHT TO 2...")
env.move_right()
dump_env(2)

print("\n>>> MOVING LEFT TO 1 (WATCH FOR DUPLICATES)...")
env.move_left()
dump_env(1)

print("\n>>> MOVING LEFT TO 0 (WATCH FOR DUPLICATES)...")
env.move_left()
dump_env(0)
