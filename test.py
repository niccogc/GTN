import torch
import torch.nn as nn
import numpy as np
import quimb.tensor as qt

# Assumes your classes are in these files or pasted above
from model.builder import Inputs
from model.NTN import NTN
torch.set_default_dtype(torch.float64)
# from ntn import NTN  <-- Assumes you import your updated NTN class here
def test_ntn_full_batch():
    print("\n" + "="*60)
    print("TESTING NTN DERIVATIVES (FULL DATASET)")
    print("="*60)

    # 1. DATA GENERATION
    torch.set_default_dtype(torch.float64)
    
    N_SAMPLES = 2000
    BATCH_SIZE = 200
    
    # Generate Data
    x_raw = 2 * torch.rand(N_SAMPLES, 1) - 1
    y_raw = x_raw**2  # Target
    x_features = torch.cat([x_raw, torch.ones_like(x_raw)], dim=1)
    
    print(f"Total Data -> X: {x_features.shape}, Y: {y_raw.shape}")

    # 2. INPUT LOADER
    input_labels = ["x1", "x2", "x3"]
    loader = Inputs(
        inputs=[x_features], 
        outputs=[y_raw],
        outputs_labels=["y"],
        input_labels=input_labels,
        batch_dim="batch",
        batch_size=BATCH_SIZE
    )

    # 3. MPS CONSTRUCTION
    D_bond = 3
    D_phys = 2
    
    def init_weights(shape):
        w = torch.randn(*shape)
        return w / torch.norm(w)

    t1 = qt.Tensor(data=init_weights((D_phys, D_bond)), inds=('x1', 'b1'), tags={'Node1'})
    t2 = qt.Tensor(data=init_weights((D_bond, D_phys, D_bond, 1)), inds=('b1', 'x2', 'b2','y'), tags={'Node2'})
    t3 = qt.Tensor(data=init_weights((D_bond, D_phys)), inds=('b2', 'x3'), tags={'Node3'})
    tn_init = qt.TensorNetwork([t1, t2, t3])

    # 4. NTN INITIALIZATION
    loss_fn = nn.MSELoss()
    model = NTN(
        tn=tn_init,
        output_dims=["y"],
        input_dims=input_labels,
        loss=loss_fn,
        data_stream=loader,
        method='cholesky'
    )
    
    # 5. FORWARD PASS (FULL DATASET)
    # We pass 'loader' directly. NTN.forward iterates over it and concatenates results.
    print(f"\nRunning Forward on full dataset ({N_SAMPLES} samples)...")

    model.fit(5, regularize = False)

if __name__ == "__main__":
    test_ntn_full_batch()
