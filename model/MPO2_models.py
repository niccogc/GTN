# type: ignore
"""
MPO2 Model Classes: MPO2, CMPO2, LMPO2, MMPO2

These classes construct tensor networks with proper node labeling to avoid
tag conflicts with input tensors created by the builder.
"""

import torch
import quimb.tensor as qt
from typing import Optional


class MPO2:
    """
    Simple MPS with output dimension.

    Structure:
    - Standard MPS chain with one site containing the output dimension
    - Tags: Node{i} for each site
    - Physical indices: x{i} for each site
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
            L: Number of sites
            bond_dim: Bond dimension for MPS
            phys_dim: Physical dimension for each site
            output_dim: Output dimension (e.g., number of classes)
            output_site: Which site gets the output dimension (default: middle site)
            init_strength: Base initialization strength before normalization (default: 0.001)
                          Only used if use_tn_normalization=False
            use_tn_normalization: Apply TN normalization after initialization (default: True)
                                 This eliminates seed-dependent collapses by normalizing outputs
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
                import numpy as np

                target_norm = np.sqrt(L * bond_dim * phys_dim)
                normalize_tn_frobenius(self.tn, target_norm=target_norm)

        self.input_labels = [f"x{i}" for i in range(L)]

        self.input_dims = [f"x{i}" for i in range(L)]

        self.output_dims = ["out"]


class CMPO2:
    """
    Cross MPO2: Two MPS layers (pixels and patches) that cross-connect.

    Structure:
    - Pixel MPS (psi): processes pixel dimensions, contains output
    - Patch MPS (phi): processes patch dimensions
    - Tags: {i}_Pi for pixel MPS nodes, {i}_Pa for patch MPS nodes
    """

    def __init__(
        self,
        L: int,
        bond_dim: int,
        phys_dim_pixels: int,
        phys_dim_patches: int,
        output_dim: int,
        output_site: Optional[int] = None,
        init_strength: float = 0.01,
    ):
        """
        Args:
            L: Number of sites
            bond_dim: Bond dimension for MPS
            phys_dim_pixels: Physical dimension for pixel MPS
            phys_dim_patches: Physical dimension for patch MPS
            output_dim: Output dimension (e.g., number of classes)
            output_site: Which site gets the output dimension (default: middle site)
            init_strength: Initialization strength for output dimension
        """
        self.L = L
        self.bond_dim = bond_dim
        self.phys_dim_pixels = phys_dim_pixels
        self.phys_dim_patches = phys_dim_patches
        self.output_dim = output_dim
        self.output_site = output_site if output_site is not None else L - 1

        # Create pixel MPS
        self.psi = qt.MPS_rand_state(L, bond_dim=bond_dim, phys_dim=phys_dim_pixels)

        # Create patch MPS
        self.phi = qt.MPS_rand_state(L, bond_dim=bond_dim, phys_dim=phys_dim_patches)

        # Add output dimension to pixel MPS at output_site
        output_node = self.psi[f"I{self.output_site}"]
        output_node.new_ind(
            "out", size=output_dim, axis=-1, mode="random", rand_strength=init_strength
        )

        # Convert to torch tensors
        self.psi.apply_to_arrays(lambda x: torch.tensor(x, dtype=torch.float32))
        self.phi.apply_to_arrays(lambda x: torch.tensor(x, dtype=torch.float32))

        # Reindex physical dimensions
        self.psi.reindex({f"k{i}": f"{i}_pixels" for i in range(L)}, inplace=True)
        self.phi.reindex({f"k{i}": f"{i}_patches" for i in range(L)}, inplace=True)

        # Replace default I{i} tags with unique tags to avoid conflict with input tensors
        for i in range(L):
            # Remove default I{i} tag and add custom tag
            psi_tensor = self.psi[f"I{i}"]
            psi_tensor.drop_tags(f"I{i}")
            psi_tensor.add_tag(f"{i}_Pi")

            phi_tensor = self.phi[f"I{i}"]
            phi_tensor.drop_tags(f"I{i}")
            phi_tensor.add_tag(f"{i}_Pa")

        # Combine into tensor network
        self.tn = self.psi & self.phi

        # Define input labels for builder
        self.input_labels = [[0, (f"{i}_patches", f"{i}_pixels")] for i in range(L)]

        # Define input_dims for NTN (simple site labels)
        self.input_dims = [str(i) for i in range(L)]

        # Define output dimensions
        self.output_dims = ["out"]


class LMPO2:
    """
    Linear MPO2: MPO for dimensionality reduction, then MPS for output.

    Structure:
    - MPO layer: input_dims → reduced_dims (trainable)
    - MPS layer: reduced_dims → output_dims (trainable)
    - Tags: {i}_MPO for MPO nodes, {i}_MPS for MPS nodes

    The reduction factor can be specified as reduced_dim/input_dim.
    For example, input_dim=10, reduced_dim=5 gives 50% reduction.
    """

    def __init__(
        self,
        L: int,
        bond_dim: int,
        phys_dim: int,
        reduced_dim: int,
        output_dim: int = 1,
        output_site: Optional[int] = None,
        init_strength: float = 0.001,
        bond_dim_mpo: int = 2,
        use_tn_normalization: bool = True,
        tn_target_std: float = 0.1,
        sample_inputs: Optional[qt.TensorNetwork] = None,
    ):
        self.L = L
        self.bond_dim = bond_dim
        self.bond_dim_mpo = bond_dim_mpo
        self.phys_dim = phys_dim
        self.input_dim = phys_dim
        self.reduced_dim = reduced_dim
        self.output_dim = output_dim
        self.output_site = output_site if output_site is not None else L - 1

        base_init = 0.1 if use_tn_normalization else init_strength

        self.mpo_tensors = []
        for i in range(L):
            if i == 0:
                data = torch.randn(phys_dim, reduced_dim, bond_dim_mpo) * base_init
                inds = (f"{i}_in", f"{i}_reduced", f"b_mpo_{i}")
                tags = {f"{i}_MPO"}
            elif i == L - 1:
                data = torch.randn(bond_dim_mpo, phys_dim, reduced_dim) * base_init
                inds = (f"b_mpo_{i - 1}", f"{i}_in", f"{i}_reduced")
                tags = {f"{i}_MPO"}
            else:
                data = torch.randn(bond_dim_mpo, phys_dim, reduced_dim, bond_dim_mpo) * base_init
                inds = (f"b_mpo_{i - 1}", f"{i}_in", f"{i}_reduced", f"b_mpo_{i}")
                tags = {f"{i}_MPO"}

            self.mpo_tensors.append(qt.Tensor(data=data, inds=inds, tags=tags))

        # Create MPS that takes reduced dimensions as input
        self.mps_tensors = []
        for i in range(L):
            if i == 0:
                data = torch.randn(reduced_dim, bond_dim) * base_init
                inds = (f"{i}_reduced", f"b_mps_{i}")
                tags = {f"{i}_MPS"}
            elif i == L - 1:
                data = torch.randn(bond_dim, reduced_dim) * base_init
                inds = (f"b_mps_{i - 1}", f"{i}_reduced")
                tags = {f"{i}_MPS"}
            else:
                data = torch.randn(bond_dim, reduced_dim, bond_dim) * base_init
                inds = (f"b_mps_{i - 1}", f"{i}_reduced", f"b_mps_{i}")
                tags = {f"{i}_MPS"}

            self.mps_tensors.append(qt.Tensor(data=data, inds=inds, tags=tags))

        # Add output dimension to MPS output site
        output_tensor = self.mps_tensors[self.output_site]
        new_inds = list(output_tensor.inds) + ["out"]
        new_shape = output_tensor.shape + (output_dim,)
        output_tensor.modify(data=torch.randn(*new_shape) * init_strength, inds=new_inds)

        self.tn = qt.TensorNetwork(self.mpo_tensors + self.mps_tensors)

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

        self.input_labels = [f"{i}_in" for i in range(L)]

        # Define input_dims for NTN (must match the physical input indices!)
        self.input_dims = [f"{i}_in" for i in range(L)]

        # Define output dimensions
        self.output_dims = ["out"]


class MMPO2:
    """
    Masking MPO2: Non-trainable MPO mask, then trainable MPS.

    Structure:
    - MPO layer: input_dims × input_dims (NOT trainable, cumulative sum mask)
    - MPS layer: input_dims → output_dims (trainable)
    - Tags: {i}_Mask for MPO nodes (with NT tag), {i}_MPS for MPS nodes

    Mask is defined as: C^{i_in, i_out}_{b_left, b_right} = sum_k H_{b_left, k} * D_{k, i_in, i_out, b_right}
    where H_{ij} = theta(j-i) (Heaviside) and D is Kronecker delta
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
        self.L = L
        self.bond_dim = bond_dim
        self.phys_dim = phys_dim
        self.input_dim = phys_dim
        self.output_dim = output_dim
        self.output_site = output_site if output_site is not None else L - 1

        base_init = 0.1 if use_tn_normalization else init_strength

        H = torch.zeros(phys_dim, phys_dim)
        for i in range(phys_dim):
            for j in range(phys_dim):
                H[i, j] = 1.0 if j >= i else 0.0

        mask_bond_dim = phys_dim
        self.mask_tensors = []

        for site_idx in range(L):
            if site_idx == 0:
                Delta = torch.zeros(phys_dim, phys_dim, mask_bond_dim)
                for k in range(phys_dim):
                    Delta[k, k, k] = 1.0
                data = Delta

                inds = (f"{site_idx}_in", f"{site_idx}_masked", f"b_mask_{site_idx}")
                tags = {f"{site_idx}_Mask", "NT"}

            elif site_idx == L - 1:
                Delta = torch.zeros(mask_bond_dim, phys_dim, phys_dim)
                for k in range(mask_bond_dim):
                    Delta[k, k, k] = 1.0

                data = torch.einsum("bk,kio->bio", H, Delta)

                inds = (f"b_mask_{site_idx - 1}", f"{site_idx}_in", f"{site_idx}_masked")
                tags = {f"{site_idx}_Mask", "NT"}

            else:
                Delta = torch.zeros(mask_bond_dim, phys_dim, phys_dim, mask_bond_dim)
                for k in range(mask_bond_dim):
                    Delta[k, k, k, k] = 1.0

                data = torch.einsum("bk,kior->bior", H, Delta)

                inds = (
                    f"b_mask_{site_idx - 1}",
                    f"{site_idx}_in",
                    f"{site_idx}_masked",
                    f"b_mask_{site_idx}",
                )
                tags = {f"{site_idx}_Mask", "NT"}

            self.mask_tensors.append(qt.Tensor(data=data, inds=inds, tags=tags))

        self.mps_tensors = []
        for i in range(L):
            if i == 0:
                data = torch.randn(phys_dim, bond_dim) * base_init
                inds = (f"{i}_masked", f"b_mps_{i}")
                tags = {f"{i}_MPS"}
            elif i == L - 1:
                data = torch.randn(bond_dim, phys_dim) * base_init
                inds = (f"b_mps_{i - 1}", f"{i}_masked")
                tags = {f"{i}_MPS"}
            else:
                data = torch.randn(bond_dim, phys_dim, bond_dim) * base_init
                inds = (f"b_mps_{i - 1}", f"{i}_masked", f"b_mps_{i}")
                tags = {f"{i}_MPS"}

            self.mps_tensors.append(qt.Tensor(data=data, inds=inds, tags=tags))

        # Add output dimension to MPS output site
        output_tensor = self.mps_tensors[self.output_site]
        new_inds = list(output_tensor.inds) + ["out"]
        new_shape = output_tensor.shape + (output_dim,)
        output_tensor.modify(data=torch.randn(*new_shape) * init_strength, inds=new_inds)

        self.tn = qt.TensorNetwork(self.mask_tensors + self.mps_tensors)

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

        self.input_labels = [f"{i}_in" for i in range(L)]

        # input_dims must match the actual TN input indices (which are {i}_in)
        self.input_dims = [f"{i}_in" for i in range(L)]

        self.output_dims = ["out"]
