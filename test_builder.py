import torch
import numpy as np
import quimb.tensor as qt
from model.builder import Inputs

def test_explicit_mapping():
    print("\n" + "="*60)
    print("TESTING EXPLICIT SOURCE MAPPING (SCENARIO C)")
    print("="*60)

    # 1. SETUP DATA
    # ---------------------------------------------------------
    N_SAMPLES = 6
    BATCH_SIZE = 2
    
    # Source A: (Batch, 5, 4) -> filled with 1.0s for easy checking
    input_A = torch.ones(N_SAMPLES, 5, 4)
    
    # Source B: (Batch, 3, 2) -> filled with 2.0s for easy checking
    input_B = torch.ones(N_SAMPLES, 3, 2) * 2.0
    
    # Target
    output_data = torch.randn(N_SAMPLES, 1)

    print(f"Input A Shape: {input_A.shape} (Values=1.0)")
    print(f"Input B Shape: {input_B.shape} (Values=2.0)")

    # 2. INITIALIZE LOADER
    # ---------------------------------------------------------
    # We want 3 Network Nodes:
    # Node 1: Uses Source A (Index 0)
    # Node 2: Uses Source A (Index 0) -> REUSE
    # Node 3: Uses Source B (Index 1)
    
    input_labels = [
        [0, ("dim_a1", "dim_a2")], # Map Source 0 to these indices
        [0, ("dim_a3", "dim_a4")], # Map Source 0 to DIFFERENT indices
        [1, ("dim_b1", "dim_b2")]  # Map Source 1 to these indices
    ]
    
    loader = Inputs(
        inputs=[input_A, input_B], 
        outputs=[output_data],
        outputs_labels=["y"],
        input_labels=input_labels, 
        batch_dim="batch",
        batch_size=BATCH_SIZE
    )
    
    print("\nLoader Initialized.")

    # 3. INSPECT BATCH
    # ---------------------------------------------------------
    mu_list, y_tensor = loader[0]
    
    print(f"\nNumber of tensors in batch: {len(mu_list)} (Expected 3)")
    
    # --- Tensor 1 (From Source A) ---
    t1 = mu_list[0]
    print(f"\nTensor 1 Indices: {t1.inds}")
    print(f"Tensor 1 Value:   {t1.data.mean()} (Expected 1.0)")
    assert t1.inds == ('batch', 'dim_a1', 'dim_a2')
    assert np.allclose(t1.data, 1.0)
    
    # --- Tensor 2 (From Source A - Reused) ---
    t2 = mu_list[1]
    print(f"\nTensor 2 Indices: {t2.inds}")
    print(f"Tensor 2 Value:   {t2.data.mean()} (Expected 1.0)")
    assert t2.inds == ('batch', 'dim_a3', 'dim_a4')
    assert np.allclose(t2.data, 1.0)
    
    # --- Tensor 3 (From Source B) ---
    t3 = mu_list[2]
    print(f"\nTensor 3 Indices: {t3.inds}")
    print(f"Tensor 3 Value:   {t3.data.mean()} (Expected 2.0)")
    assert t3.inds == ('batch', 'dim_b1', 'dim_b2')
    assert np.allclose(t3.data, 2.0)
    
    # Check shape correctness
    assert t1.shape == (BATCH_SIZE, 5, 4)
    assert t3.shape == (BATCH_SIZE, 3, 2)

    print("\n[SUCCESS] All mappings correct.")

if __name__ == "__main__":
    test_explicit_mapping()
