import torch
import time

torch.set_default_dtype(torch.float64)
m = 10000
n = 122
# vectors = [torch.rand(m) for _ in range(n)]  # list of column vectors
r = 8  # desired rank
U_basis, _ = torch.linalg.qr(torch.randn(m, r))  # r orthonormal basis vectors
coeffs = torch.randn(r, n)
vectors = [U_basis @ coeffs[:, j] for j in range(n)]
print(vectors[0].shape)

Q = torch.zeros(m, n)
R = torch.zeros(n, n)

print("Tracking SVD during incremental Gram–Schmidt:")
svdrec = []
svd = []
start_inc = time.time()
for j, v in enumerate(vectors):
    # vectorized projection onto previous columns
    if j > 0:
        R[:j, j] = Q[:, :j].T @ v        # projections
        w = v - Q[:, :j] @ R[:j, j]      # remove previous components
    else:
        w = v.clone()
    
    # normalize
    norm_w = w.norm()
    if norm_w > 1e-12:
        R[j, j] = norm_w
        Q[:, j] = w / R[j, j]
    else:
        # vector is linearly dependent, set column to zeros
        R[j, j] = 0.0
        Q[:, j] = 0.0
    
    print(f" norm of the vector is {R[j,j]}")
    # incremental SVD of the current partial R
    U_r, S_r, V_r = torch.linalg.svd(R[:j+1, :j+1])

    # U_partial, S_partial, V_partial = torch.linalg.svd(Q[:, :j+1] @ R[:j+1, :j+1])

    print(f"column {j+1}: sum of svds = {S_r.sum():.6f}, with Q: ")
    svdrec.append(S_r.sum())
end_inc = time.time()
print(f"Incremental Gram–Schmidt + small SVD time: {end_inc - start_inc:.6f} s")
# Full matrix SVD for comparison
A = torch.stack(vectors, dim=1)
start_full = time.time()
# U_full, S_full, V_full = torch.linalg.svd(A)
for i in range(1,A.shape[-1]+1):
    U_full, S_full, V_full = torch.linalg.svd(A[:,:i])
    # print(f"\nFull SVD singular values of A: {S_full.sum()}")
    svd.append(S_full.sum())
end_full = time.time()
print(f"Full SVD time: {end_full - start_full:.6f} s")

# Reconstruction check
reconstruction_error = torch.norm(A - Q @ R)
print("Reconstruction error:", reconstruction_error.item())

print((torch.tensor(svdrec) - torch.tensor(svd)).sum())
