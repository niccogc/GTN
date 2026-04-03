# type: ignore
"""
CPDA (Canonical Polyadic Decomposition Asymmetric) Model Class

CPDA Structure:
- L factor matrices (nodes), each with shape (phys_dim, bond_dim)
- All nodes share a bond_dim index "r" (unlike MPS/MPO which use bond indices between neighbors)
- One node (output_site) has additional output index: (phys_dim, bond_dim, out)
- Contraction produces output via elementwise multiplication over the bond_dim dimension
"""

import torch
import quimb.tensor as qt
from typing import Optional


class CPDA:
    """
    Canonical Polyadic Decomposition Asymmetric tensor network.

    Structure:
         x0    x1    x2    ...   xL-1
         |     |     |           |
      [Node0][Node1][Node2]...[NodeL-1]
         \\     |     |           /
          \\    |    /           /
           \\   |   /           /
            \\  |  /           /
             [r]  (shared bond_dim index)
              |
            [out] (output at output_site node)

    Each node has shape (phys_dim, bond_dim) except the output node which has
    shape (phys_dim, bond_dim, output_dim).

    Unlike MPS where bond indices connect adjacent nodes, in CPDA all nodes
    share the same bond_dim index 'r'. The final output is computed by contracting
    all physical indices with inputs and summing over the shared bond_dim dimension.
    """

    def __init__(
        self,
        L: int,
        bond_dim: int,
        phys_dim: int,
        output_dim: int,
        output_site: Optional[int] = None,
        init_strength: float = 0.001,
        use_tn_normalization: bool = True,
        tn_target_std: float = 0.1,
        sample_inputs: Optional[qt.TensorNetwork] = None,
    ):
        """
        Args:
            L: Number of sites/features
            bond_dim: CPD rank (number of components)
            phys_dim: Physical dimension for each site
            output_dim: Output dimension (e.g., number of classes)
            output_site: Which site gets the output dimension (default: last site)
            init_strength: Base initialization strength (default: 0.001)
                          Only used if use_tn_normalization=False
            use_tn_normalization: Apply TN normalization after initialization (default: True)
            tn_target_std: Target standard deviation for TN normalization (default: 0.1)
            sample_inputs: Sample TN inputs for normalization. If None and use_tn_normalization=True,
                          will use Frobenius norm normalization instead
        """
        self.L = L
        self.bond_dim = bond_dim
        self.phys_dim = phys_dim
        self.output_dim = output_dim
        self.output_site = output_site if output_site is not None else L - 1

        base_init = 0.1 if use_tn_normalization else init_strength

        tensors = []
        
        # Special case for L=1: no shared rank dimension needed
        if L == 1:
            shape = (phys_dim, output_dim)
            inds = ("x0", "out")
            data = torch.randn(*shape) * base_init
            tensor = qt.Tensor(data=data, inds=inds, tags={"Node0"})
            tensors.append(tensor)
        else:
            for i in range(L):
                if i == self.output_site:
                    # Output node: (phys_dim, bond_dim, output_dim)
                    shape = (phys_dim, bond_dim, output_dim)
                    inds = (f"x{i}", "r", "out")
                else:
                    # Regular node: (phys_dim, bond_dim)
                    shape = (phys_dim, bond_dim)
                    inds = (f"x{i}", "r")

                data = torch.randn(*shape) * base_init
                tensor = qt.Tensor(data=data, inds=inds, tags={f"Node{i}"})
                tensors.append(tensor)

        self.tn = qt.TensorNetwork(tensors)

        if use_tn_normalization:
            from model.initialization import normalize_tn_output, normalize_tn_frobenius

            if sample_inputs is not None:
                normalize_tn_output(
                    self.tn,
                    sample_inputs,
                    output_dims=["out"],
                    batch_dim="s",
                    target_std=tn_target_std,
                )
            else:
                import numpy as np

                target_norm = np.sqrt(L * bond_dim * phys_dim)
                normalize_tn_frobenius(self.tn, target_norm=target_norm)

        self.input_labels = [f"x{i}" for i in range(L)]
        self.input_dims = [f"x{i}" for i in range(L)]
        self.output_dims = ["out"]
