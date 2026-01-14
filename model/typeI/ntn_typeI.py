# type: ignore
"""
Type I Model Builders: MPO2TypeI, LMPO2TypeI, MMPO2TypeI

These classes construct ensembles of tensor networks with varying number of sites (L=1,2,...,max_sites).
They expose the same interface pattern as standard models but with lists:
- .tns: List of TensorNetworks
- .input_dims_list: List of input_dims for each TN
- .input_labels_list: List of input_labels for each TN
- .output_dims: Output dimensions (same for all TNs)
"""

import torch
import quimb.tensor as qt
from typing import List, Optional


def _create_simple_mps(
    L: int,
    bond_dim: int,
    phys_dim: int,
    output_dim: int,
    output_site: Optional[int] = None,
    init_strength: float = 0.001,
):
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
    input_dims = input_labels
    output_dims = ["out"]

    return tn, input_labels, input_dims, output_dims


def _create_lmpo2(
    L: int,
    bond_dim: int,
    phys_dim: int,
    reduced_dim: int,
    output_dim: int,
    output_site: Optional[int] = None,
    init_strength: float = 0.001,
):
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
    input_dims = input_labels
    output_dims = ["out"]

    return tn, input_labels, input_dims, output_dims


def _create_mmpo2(
    L: int,
    bond_dim: int,
    phys_dim: int,
    output_dim: int,
    output_site: Optional[int] = None,
    init_strength: float = 0.001,
):
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
    input_dims = input_labels
    output_dims = ["out"]

    return tn, input_labels, input_dims, output_dims


class MPO2TypeI:
    def __init__(
        self,
        max_sites: int,
        bond_dim: int,
        phys_dim: int,
        output_dim: int,
        output_site: Optional[int] = None,
        init_strength: float = 0.001,
    ):
        self.max_sites = max_sites
        self.bond_dim = bond_dim
        self.phys_dim = phys_dim
        self.output_dim = output_dim

        self.tns: List[qt.TensorNetwork] = []
        self.input_dims_list: List[List[str]] = []
        self.input_labels_list: List[List[str]] = []
        self.output_dims = ["out"]

        for L in range(1, max_sites + 1):
            tn, input_labels, input_dims, _ = _create_simple_mps(
                L=L,
                bond_dim=bond_dim,
                phys_dim=phys_dim,
                output_dim=output_dim,
                output_site=output_site,
                init_strength=init_strength,
            )
            self.tns.append(tn)
            self.input_dims_list.append(input_dims)
            self.input_labels_list.append(input_labels)


class LMPO2TypeI:
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
        self.max_sites = max_sites
        self.bond_dim = bond_dim
        self.phys_dim = phys_dim
        self.reduced_dim = reduced_dim
        self.output_dim = output_dim

        self.tns: List[qt.TensorNetwork] = []
        self.input_dims_list: List[List[str]] = []
        self.input_labels_list: List[List[str]] = []
        self.output_dims = ["out"]

        for L in range(1, max_sites + 1):
            tn, input_labels, input_dims, _ = _create_lmpo2(
                L=L,
                bond_dim=bond_dim,
                phys_dim=phys_dim,
                reduced_dim=reduced_dim,
                output_dim=output_dim,
                output_site=output_site,
                init_strength=init_strength,
            )
            self.tns.append(tn)
            self.input_dims_list.append(input_dims)
            self.input_labels_list.append(input_labels)


class MMPO2TypeI:
    def __init__(
        self,
        max_sites: int,
        bond_dim: int,
        phys_dim: int,
        output_dim: int,
        output_site: Optional[int] = None,
        init_strength: float = 0.001,
    ):
        self.max_sites = max_sites
        self.bond_dim = bond_dim
        self.phys_dim = phys_dim
        self.output_dim = output_dim

        self.tns: List[qt.TensorNetwork] = []
        self.input_dims_list: List[List[str]] = []
        self.input_labels_list: List[List[str]] = []
        self.output_dims = ["out"]
        self.not_trainable_tags = ["NT"]

        for L in range(1, max_sites + 1):
            tn, input_labels, input_dims, _ = _create_mmpo2(
                L=L,
                bond_dim=bond_dim,
                phys_dim=phys_dim,
                output_dim=output_dim,
                output_site=output_site,
                init_strength=init_strength,
            )
            self.tns.append(tn)
            self.input_dims_list.append(input_dims)
            self.input_labels_list.append(input_labels)
