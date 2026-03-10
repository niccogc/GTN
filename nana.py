import numpy as np

# Dimensions (S >= I*J for H to be invertible)
I, J, S = 2, 2, 4  

L = np.random.rand(S, I)
E = np.random.rand(S, J)

# Construct A where A_{s, (ij)} = L_{si} * E_{sj}
# Using broadcasting to create all (ij) combinations for each s
A = (L[:, :, None] * E[:, None, :]).reshape(S, I * J)

# Compute H = A^T @ A
H = A.T @ A

# Compute inverses
H_inv = np.linalg.inv(H)
A_inv = np.linalg.inv(A)

# Theoretical component-wise inverse: (A^-1) @ (A^-1)^T
# This corresponds to the sum over s: (A^-1)_{ij, s} * (A^-1)_{i'j', s}
H_inv_explicit = A_inv @ A_inv.T

# Proof
matches = np.allclose(H_inv, H_inv_explicit)
print(f"Mathematical identity holds: {matches}")
