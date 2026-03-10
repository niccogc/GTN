import itertools
from collections import defaultdict
import numpy as np
import quimb.tensor as qt

def base(dim, n):
    A = np.zeros(dim)
    A[int(n)] = 1
    return A

def cp_vecs_from_coord(coord, dim):
    # Use 'dim' instead of len(coord) to define vector size
    return [base(dim, i) for i in coord]

def cp_from_coord_list(coordinates, dim):
    # Pass 'dim' down to the vector generator
    all_vec_sets = [cp_vecs_from_coord(c, dim) for c in coordinates]
    stacked_vecs = np.stack(all_vec_sets, axis=1)
    return stacked_vecs

def find_s_simple(D, N):
    # Create the starting pool: e.g., if D=[1,2] and N=3, pool is [1, 2, 0]
    num_zeros = N - len(D)
    pool = list(D) + [0] * num_zeros
    
    # Get all unique permutations of this pool
    return set(itertools.permutations(pool, N))

dim = 6
D = [1,1]
N = 4
coords = find_s_simple(D, N)
print(coords)

CP = cp_from_coord_list(coords, dim=dim)
print("CPSHAPE")

tnlist = [qt.Tensor(i, inds=["inp", "s"]) for i in CP]
tn = qt.TensorNetwork(tnlist)

print(tn)
