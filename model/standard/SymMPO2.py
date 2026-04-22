# type: ignore
"""
Symmetric MPO2 Model: A-B-A structure where A nodes are tied (same tensor).

For L=3: Node0 (A) - Node1 (B) - Node2 (A, same as Node0)
The model uses virtual copies - only stores A and B, but constructs
full TN with proper indices for contraction.
"""

import torch
import quimb.tensor as qt
from typing import Optional, List


class SymMPO2:
    """
    Symmetric MPO2 with A-B-A structure.

    Structure for L=3:
        L (x0, b0) -- B (b0, x1, b1, out) -- L (b1, x2)
    It only needs to be passed L, and B.
    as self.Left [L0 L1 ..] and self.Central [B]
    Both L nodes share the same parameters.
    B is at center, contains output dimension.

    For general odd L:
        L0 - L1 - ... - B - ... - L1 - L0

    Symmetric nodes are tied via `sym_groups` dict.
    """

    def __init__(
        self,
        L: int,
        bond_dim: int,
        phys_dim: int,
        output_dim: int,
        init_strength: float = 0.1,
        use_tn_normalization: bool = True,
        tn_target_std: float = 0.1,
    ):
        """
        Args:
            L: Number of sites (must be odd for symmetric structure)
            bond_dim: Bond dimension
            phys_dim: Physical dimension for each site
            output_dim: Output dimension
            init_strength: Initialization strength
        """
        if L < 3:
            raise ValueError("SymMPO2 requires L >= 3")

        self.L = L
        self.bond_dim = bond_dim
        self.phys_dim = phys_dim
        self.output_dim = output_dim
        if self.L%2 == 0:
            print("KEEP L ODD PLEASE DEAR LORD")

        base_init = 0.1 if use_tn_normalization else init_strength
        tensors = []
        if L == 1:
            shape = (phys_dim, output_dim)
            inds = ("x0", "out")
            data = torch.randn(*shape) * base_init
            tensor = qt.Tensor(data=data, inds=inds, tags={"Node0"})
            tensors.append(tensor)
        else:
            for i in range(L//2):
                if i == 0:
                    shape = (phys_dim, bond_dim)
                    inds = (f"x{i}", f"b{i}")
                else:
                    shape = (bond_dim, phys_dim, bond_dim)
                    inds = (f"b{i - 1}", f"x{i}", f"b{i}")

                data = torch.randn(*shape) * base_init
                tensor = qt.Tensor(data=data, inds=inds, tags={f"L{i}"})
                tensors.append(tensor)

        self.Left = qt.TensorNetwork(tensors)

        shape = (bond_dim, phys_dim, output_dim, bond_dim)
        inds = (f"b{L//2 - 1}", f"x{L//2}", "out", f"b{L//2-1}_prime")
        sym_tensor = torch.randn(*shape) * base_init
        sym_tensor = 0.5 * (sym_tensor + sym_tensor.transpose(0, 3))
        self.Central = qt.Tensor(data= sym_tensor, inds=inds, tags=["C"])

        if use_tn_normalization:
            from model.initialization import normalize_tn_frobenius

            target_norm = torch.sqrt(torch.tensor(L * bond_dim * phys_dim))
            normalize_tn_frobenius(self.Left, target_norm=target_norm)

        self.input_labels = [f"x{i}" for i in range(L//2 + 1)]
        self.input_dims = [f"x{i}" for i in range(L//2 + 1)]
        self.output_dims = ["out"]
