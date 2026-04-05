# type: ignore
"""TNML_P: MPS with phys_dim=L+1 (polynomial), TNML_F: MPS with phys_dim=2 (Fourier)."""

import torch
import quimb.tensor as qt
import numpy as np
from typing import Optional


class TNML_P:
    """MPS with one node per feature, polynomial basis: [1, x_i, x_i^2, ..., x_i^L]."""

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
        self.L = L
        self.poly_degree = L
        self.n_features = phys_dim
        self.bond_dim = bond_dim
        self.phys_dim = L + 1
        self.output_dim = output_dim
        self.output_site = output_site if output_site is not None else phys_dim - 1

        base_init = 0.1 if use_tn_normalization else init_strength
        n_sites = phys_dim

        tensors = []
        if n_sites == 1:
            shape = (self.phys_dim, output_dim)
            inds = ("x0", "out")
            data = torch.randn(*shape) * base_init
            tensor = qt.Tensor(data=data, inds=inds, tags={"Node0"})
            tensors.append(tensor)
        else:
            for i in range(n_sites):
                if i == 0:
                    shape = (self.phys_dim, bond_dim)
                    inds = (f"x{i}", f"b{i}")
                elif i == n_sites - 1:
                    shape = (bond_dim, self.phys_dim)
                    inds = (f"b{i - 1}", f"x{i}")
                else:
                    shape = (bond_dim, self.phys_dim, bond_dim)
                    inds = (f"b{i - 1}", f"x{i}", f"b{i}")

                if i == self.output_site:
                    shape = shape + (output_dim,)
                    inds = inds + ("out",)

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
                target_norm = np.sqrt(n_sites * bond_dim * self.phys_dim)
                normalize_tn_frobenius(self.tn, target_norm=target_norm)

        self.input_labels = [f"x{i}" for i in range(n_sites)]
        self.input_dims = [f"x{i}" for i in range(n_sites)]
        self.output_dims = ["out"]
        self.encoding = "polynomial"


class TNML_F:
    """MPS with one node per feature, Fourier basis: [cos(x_i*pi/2), sin(x_i*pi/2)]."""

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
        self.L = L
        self.n_features = phys_dim
        self.bond_dim = bond_dim
        self.phys_dim = 2
        self.output_dim = output_dim
        self.output_site = output_site if output_site is not None else phys_dim - 1

        base_init = 0.1 if use_tn_normalization else init_strength
        n_sites = phys_dim

        tensors = []
        if n_sites == 1:
            shape = (self.phys_dim, output_dim)
            inds = ("x0", "out")
            data = torch.randn(*shape) * base_init
            tensor = qt.Tensor(data=data, inds=inds, tags={"Node0"})
            tensors.append(tensor)
        else:
            for i in range(n_sites):
                if i == 0:
                    shape = (self.phys_dim, bond_dim)
                    inds = (f"x{i}", f"b{i}")
                elif i == n_sites - 1:
                    shape = (bond_dim, self.phys_dim)
                    inds = (f"b{i - 1}", f"x{i}")
                else:
                    shape = (bond_dim, self.phys_dim, bond_dim)
                    inds = (f"b{i - 1}", f"x{i}", f"b{i}")

                if i == self.output_site:
                    shape = shape + (output_dim,)
                    inds = inds + ("out",)

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
                target_norm = np.sqrt(n_sites * bond_dim * self.phys_dim)
                normalize_tn_frobenius(self.tn, target_norm=target_norm)

        self.input_labels = [f"x{i}" for i in range(n_sites)]
        self.input_dims = [f"x{i}" for i in range(n_sites)]
        self.output_dims = ["out"]
        self.encoding = "fourier"
