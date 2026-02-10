import torch

def test_structural_equivalence():
    # Dimensions
    n_v = 10  # Space dimension (i)
    n_a = 4   # Model dimension (j)
    
    # Random problem parameters
    v = torch.randn(n_v)
    A = torch.randn(n_a, n_a)  # The a_{j', j} matrix
    b_vec = torch.randn(n_a)   # The a_{j'} vector on RHS
    
    # 1. CONSTRUCT THE FULL KRONECKER SYSTEM
    # The LHS matrix is (v v^T) \otimes A
    # The RHS is v \otimes b_vec
    LHS_full = torch.kron(torch.outer(v, v), A)
    RHS_full = torch.kron(v, b_vec)
    
    # 2. APPLY YOUR "TRICK" (Structural Solve)
    # Target Equation: (v^T v) * A @ delta_collapsed = RHS_collapsed
    # We multiply both sides of the full system by (v.T / ||v||^2 \otimes I)
    
    v_inv = v / torch.sum(v**2)
    operator = v_inv.unsqueeze(0) # Row vector (1, n_v)
    
    # Collapsing the RHS: (v_inv @ v) * b_vec = b_vec
    RHS_collapsed = b_vec 
    
    # Collapsing the LHS: (v_inv @ v @ v.T) \otimes A = v.T \otimes A
    # To isolate delta, we recognize delta_ij = v_i * gamma_j
    # This leads to: A @ gamma = b_vec
    gamma = torch.linalg.solve(A, b_vec)
    delta_structural = torch.outer(v_inv, gamma)
    
    # 3. VERIFICATION
    # Check if this delta satisfies the HUGE original system: LHS_full @ delta = RHS_full
    delta_flat = delta_structural.T.reshape(-1) # Aligning indices for Kronecker
    
    # Note: Depending on your Kronecker ordering (i,j) or (j,i), 
    # we may need to flatten carefully.
    # For sum_{i,j} (v_i' v_i) a_{j'j} delta_{ij}, delta should be flattened 
    # such that j is the faster index.
    delta_flat = delta_structural.reshape(-1) 
    
    actual_RHS = LHS_full @ delta_flat
    error = torch.norm(actual_RHS - RHS_full)
    
    print(f"--- Dimension Analysis ---")
    print(f"Full System Size: {LHS_full.shape[0]}x{LHS_full.shape[1]}")
    print(f"Reduced System Size: {n_a}x{n_a}")
    print(f"\nVerification Error: {error.item():.2e}")
    
    if error < 1e-5:
        print("SUCCESS: The structural solve satisfies the full Kronecker problem.")

if __name__ == "__main__":
    test_structural_equivalence()
