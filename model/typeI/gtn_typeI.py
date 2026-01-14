# type: ignore
"""
GTN Type I: Ensemble of GTN models with varying number of sites.

For GTN, the ensemble is simpler than NTN because PyTorch autograd
handles the gradient computation automatically when we sum the outputs.
"""

import torch
import torch.nn as nn
import quimb.tensor as qt
from typing import List, Optional

from model.base.GTN import GTN


def create_simple_mps_tn(
    L: int,
    bond_dim: int,
    phys_dim: int,
    output_dim: int,
    output_site: Optional[int] = None,
    init_strength: float = 0.001,
):
    """Create simple MPS tensor network for GTN."""
    if output_site is None:
        output_site = L - 1

    tensors = []

    if L == 1:
        shape = (phys_dim, output_dim)
        inds = ("x0", "out")
        data = torch.randn(*shape) * init_strength
        tensor = qt.Tensor(data=data, inds=inds, tags={"Node0"})
        tensors.append(tensor)
    else:
        for i in range(L):
            if i == 0:
                shape = (phys_dim, bond_dim)
                inds = (f"x{i}", f"b{i}")
            elif i == L - 1:
                shape = (bond_dim, phys_dim)
                inds = (f"b{i - 1}", f"x{i}")
            else:
                shape = (bond_dim, phys_dim, bond_dim)
                inds = (f"b{i - 1}", f"x{i}", f"b{i}")

            if i == output_site:
                shape = shape + (output_dim,)
                inds = inds + ("out",)

            data = torch.randn(*shape) * init_strength
            tensor = qt.Tensor(data=data, inds=inds, tags={f"Node{i}"})
            tensors.append(tensor)

    tn = qt.TensorNetwork(tensors)
    input_labels = [f"x{i}" for i in range(L)]
    output_dims = ["out"]

    return tn, input_labels, output_dims


def create_lmpo2_tn(
    L: int,
    bond_dim: int,
    phys_dim: int,
    reduced_dim: int,
    output_dim: int,
    output_site: Optional[int] = None,
    init_strength: float = 0.001,
):
    """Create LMPO2 tensor network for GTN."""
    if output_site is None:
        output_site = L - 1

    tensors = []

    if L == 1:
        mpo_data = torch.randn(phys_dim, reduced_dim) * init_strength
        mpo_tensor = qt.Tensor(data=mpo_data, inds=("0_in", "0_reduced"), tags={"0_MPO"})
        tensors.append(mpo_tensor)

        mps_data = torch.randn(reduced_dim, output_dim) * init_strength
        mps_tensor = qt.Tensor(data=mps_data, inds=("0_reduced", "out"), tags={"0_MPS"})
        tensors.append(mps_tensor)
    else:
        for i in range(L):
            if i == 0:
                data = torch.randn(phys_dim, reduced_dim, bond_dim) * init_strength
                inds = (f"{i}_in", f"{i}_reduced", f"b_mpo_{i}")
            elif i == L - 1:
                data = torch.randn(bond_dim, phys_dim, reduced_dim) * init_strength
                inds = (f"b_mpo_{i - 1}", f"{i}_in", f"{i}_reduced")
            else:
                data = torch.randn(bond_dim, phys_dim, reduced_dim, bond_dim) * init_strength
                inds = (f"b_mpo_{i - 1}", f"{i}_in", f"{i}_reduced", f"b_mpo_{i}")
            tensors.append(qt.Tensor(data=data, inds=inds, tags={f"{i}_MPO"}))

        for i in range(L):
            if i == 0:
                shape = (reduced_dim, bond_dim)
                inds = (f"{i}_reduced", f"b_mps_{i}")
            elif i == L - 1:
                shape = (bond_dim, reduced_dim)
                inds = (f"b_mps_{i - 1}", f"{i}_reduced")
            else:
                shape = (bond_dim, reduced_dim, bond_dim)
                inds = (f"b_mps_{i - 1}", f"{i}_reduced", f"b_mps_{i}")

            if i == output_site:
                shape = shape + (output_dim,)
                inds = inds + ("out",)

            data = torch.randn(*shape) * init_strength
            tensors.append(qt.Tensor(data=data, inds=inds, tags={f"{i}_MPS"}))

    tn = qt.TensorNetwork(tensors)
    input_labels = [f"{i}_in" for i in range(L)]
    output_dims = ["out"]

    return tn, input_labels, output_dims


def create_mmpo2_tn(
    L: int,
    bond_dim: int,
    phys_dim: int,
    output_dim: int,
    output_site: Optional[int] = None,
    init_strength: float = 0.001,
):
    """Create MMPO2 tensor network for GTN (mask is non-trainable)."""
    if output_site is None:
        output_site = L - 1

    tensors = []
    mask_bond_dim = phys_dim

    H = torch.zeros(phys_dim, phys_dim)
    for i in range(phys_dim):
        for j in range(phys_dim):
            H[i, j] = 1.0 if j >= i else 0.0

    if L == 1:
        mask_data = torch.eye(phys_dim)
        mask_tensor = qt.Tensor(data=mask_data, inds=("0_in", "0_masked"), tags={"0_Mask", "NT"})
        tensors.append(mask_tensor)

        mps_data = torch.randn(phys_dim, output_dim) * init_strength
        mps_tensor = qt.Tensor(data=mps_data, inds=("0_masked", "out"), tags={"0_MPS"})
        tensors.append(mps_tensor)
    else:
        for i in range(L):
            if i == 0:
                Delta = torch.zeros(phys_dim, phys_dim, mask_bond_dim)
                for k in range(phys_dim):
                    Delta[k, k, k] = 1.0
                data = Delta
                inds = (f"{i}_in", f"{i}_masked", f"b_mask_{i}")
            elif i == L - 1:
                Delta = torch.zeros(mask_bond_dim, phys_dim, phys_dim)
                for k in range(mask_bond_dim):
                    Delta[k, k, k] = 1.0
                data = torch.einsum("bk,kio->bio", H, Delta)
                inds = (f"b_mask_{i - 1}", f"{i}_in", f"{i}_masked")
            else:
                Delta = torch.zeros(mask_bond_dim, phys_dim, phys_dim, mask_bond_dim)
                for k in range(mask_bond_dim):
                    Delta[k, k, k, k] = 1.0
                data = torch.einsum("bk,kior->bior", H, Delta)
                inds = (f"b_mask_{i - 1}", f"{i}_in", f"{i}_masked", f"b_mask_{i}")
            tensors.append(qt.Tensor(data=data, inds=inds, tags={f"{i}_Mask", "NT"}))

        for i in range(L):
            if i == 0:
                shape = (phys_dim, bond_dim)
                inds = (f"{i}_masked", f"b_mps_{i}")
            elif i == L - 1:
                shape = (bond_dim, phys_dim)
                inds = (f"b_mps_{i - 1}", f"{i}_masked")
            else:
                shape = (bond_dim, phys_dim, bond_dim)
                inds = (f"b_mps_{i - 1}", f"{i}_masked", f"b_mps_{i}")

            if i == output_site:
                shape = shape + (output_dim,)
                inds = inds + ("out",)

            data = torch.randn(*shape) * init_strength
            tensors.append(qt.Tensor(data=data, inds=inds, tags={f"{i}_MPS"}))

    tn = qt.TensorNetwork(tensors)
    input_labels = [f"{i}_in" for i in range(L)]
    output_dims = ["out"]

    return tn, input_labels, output_dims


class GTN_TypeI_Model(GTN):
    """GTN model for TypeI ensemble - customizes construct_nodes per model."""

    def __init__(self, input_labels, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._input_labels = input_labels

    def construct_nodes(self, x):
        input_nodes = []
        for label in self._input_labels:
            node = qt.Tensor(x, inds=["s", label], tags=f"Input_{label}")
            input_nodes.append(node)
        return input_nodes


class MPO2TypeI_GTN(nn.Module):
    """Type I ensemble of simple MPS models using GTN (autograd)."""

    def __init__(
        self,
        max_sites: int,
        bond_dim: int,
        phys_dim: int,
        output_dim: int,
        output_site: Optional[int] = None,
        init_strength: float = 0.001,
    ):
        super().__init__()
        self.max_sites = max_sites
        self.models = nn.ModuleList()

        for L in range(1, max_sites + 1):
            tn, input_labels, output_dims = create_simple_mps_tn(
                L=L,
                bond_dim=bond_dim,
                phys_dim=phys_dim,
                output_dim=output_dim,
                output_site=output_site,
                init_strength=init_strength,
            )
            model = GTN_TypeI_Model(
                input_labels=input_labels,
                tn=tn,
                output_dims=output_dims,
                input_dims=input_labels,
            )
            self.models.append(model)

    def forward(self, x):
        total = None
        for model in self.models:
            y = model(x)
            if total is None:
                total = y
            else:
                total = total + y
        return total


class LMPO2TypeI_GTN(nn.Module):
    """Type I ensemble of LMPO2 models using GTN (autograd)."""

    def __init__(
        self,
        max_sites: int,
        bond_dim: int,
        phys_dim: int,
        reduced_dim: int,
        output_dim: int,
        output_site: Optional[int] = None,
        init_strength: float = 0.001,
    ):
        super().__init__()
        self.max_sites = max_sites
        self.models = nn.ModuleList()

        for L in range(1, max_sites + 1):
            tn, input_labels, output_dims = create_lmpo2_tn(
                L=L,
                bond_dim=bond_dim,
                phys_dim=phys_dim,
                reduced_dim=reduced_dim,
                output_dim=output_dim,
                output_site=output_site,
                init_strength=init_strength,
            )
            model = GTN_TypeI_Model(
                input_labels=input_labels,
                tn=tn,
                output_dims=output_dims,
                input_dims=input_labels,
            )
            self.models.append(model)

    def forward(self, x):
        total = None
        for model in self.models:
            y = model(x)
            if total is None:
                total = y
            else:
                total = total + y
        return total


class MMPO2TypeI_GTN(nn.Module):
    """Type I ensemble of MMPO2 models using GTN (autograd)."""

    def __init__(
        self,
        max_sites: int,
        bond_dim: int,
        phys_dim: int,
        output_dim: int,
        output_site: Optional[int] = None,
        init_strength: float = 0.001,
    ):
        super().__init__()
        self.max_sites = max_sites
        self.models = nn.ModuleList()

        for L in range(1, max_sites + 1):
            tn, input_labels, output_dims = create_mmpo2_tn(
                L=L,
                bond_dim=bond_dim,
                phys_dim=phys_dim,
                output_dim=output_dim,
                output_site=output_site,
                init_strength=init_strength,
            )
            model = GTN_TypeI_Model(
                input_labels=input_labels,
                tn=tn,
                output_dims=output_dims,
                input_dims=input_labels,
                not_trainable_tags=["NT"],
            )
            self.models.append(model)

    def forward(self, x):
        total = None
        for model in self.models:
            y = model(x)
            if total is None:
                total = y
            else:
                total = total + y
        return total
