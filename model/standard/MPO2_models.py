# type: ignore
"""
MPO2 Model Classes: MPO2, CMPO2, LMPO2, MMPO2

These classes construct tensor networks with proper node labeling to avoid
tag conflicts with input tensors created by the builder.
"""

import numpy as np
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
    
    Quimb optimal contraction order:
      Node0 [b0, x0]
        ⊗ I0 [s, x0]
        contract [x0]
        → i0 [b0, s]
      Node1 [b0, b1, x1]
        ⊗ I1 [s, x1]
        contract [x1]
        → i1 [b0, b1, s]
      Node2 [b1, out, x2]
        ⊗ I2 [s, x2]
        contract [x2]
        → i2 [b1, out, s]
      i0 [b0, s]
        ⊗ i1 [b0, b1, s]
        contract [b0]
        → i3 [b1, s]
      i3 [b1, s]
        ⊗ i2 [b1, out, s]
        contract [b1]
        → result [out, s]
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
            output_site: Which site gets the output dimension (default: last site)
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
        
        # Special case for L=1: no bond dimensions needed
        if L == 1:
            shape = (phys_dim, output_dim)
            inds = ("x0", "out")
            data = torch.randn(*shape) * base_init
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

                if i == self.output_site:
                    shape = shape + (output_dim,)
                    inds = inds + ("out",)

                data = torch.randn(*shape) * base_init
                data = data/torch.norm(data)
                tensor = qt.Tensor(data=data, inds=inds, tags={f"Node{i}"})
                tensors.append(tensor)

        self.tn = qt.TensorNetwork(tensors)
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
    
    Quimb optimal contraction order (L=3):
      0_MPO [0_in, 0_reduced, b_mpo_0]
        ⊗ I0 [0_in, s]
        contract [0_in]
        → i0 [0_reduced, b_mpo_0, s]
      1_MPO [1_in, 1_reduced, b_mpo_0, b_mpo_1]
        ⊗ I1 [1_in, s]
        contract [1_in]
        → i1 [1_reduced, b_mpo_0, b_mpo_1, s]
      1_MPS [1_reduced, b_mps_0, b_mps_1]
        ⊗ 2_MPS [2_reduced, b_mps_1, out]
        contract [b_mps_1]
        → i2 [1_reduced, 2_reduced, b_mps_0, out]
      2_MPO [2_in, 2_reduced, b_mpo_1]
        ⊗ I2 [2_in, s]
        contract [2_in]
        → i3 [2_reduced, b_mpo_1, s]
      i2 [1_reduced, 2_reduced, b_mps_0, out]
        ⊗ 0_MPS [0_reduced, b_mps_0]
        contract [b_mps_0]
        → i4 [0_reduced, 1_reduced, 2_reduced, out]
      i4 [0_reduced, 1_reduced, 2_reduced, out]
        ⊗ i1 [1_reduced, b_mpo_0, b_mpo_1, s]
        contract [1_reduced]
        → i5 [0_reduced, 2_reduced, b_mpo_0, b_mpo_1, out, s]
      i5 [0_reduced, 2_reduced, b_mpo_0, b_mpo_1, out, s]
        ⊗ i3 [2_reduced, b_mpo_1, s]
        contract [2_reduced, b_mpo_1]
        → i6 [0_reduced, b_mpo_0, out, s]
      i6 [0_reduced, b_mpo_0, out, s]
        ⊗ i0 [0_reduced, b_mpo_0, s]
        contract [0_reduced, b_mpo_0]
        → result [out, s]
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
        self.reduction_factor = reduced_dim / phys_dim
        self.output_dim = output_dim
        self.output_site = output_site if output_site is not None else L - 1

        base_init = 0.1 if use_tn_normalization else init_strength

        tensors = []

        # Special case for L=1: no bond dimensions needed
        if L == 1:
            mpo_data = torch.randn(phys_dim, reduced_dim) * base_init
            mpo_data = mpo_data / torch.norm(mpo_data)
            mpo_tensor = qt.Tensor(data=mpo_data, inds=("0_in", "0_reduced"), tags={"0_MPO"})
            tensors.append(mpo_tensor)

            mps_data = torch.randn(reduced_dim, output_dim) * base_init
            mps_data = mps_data / torch.norm(mps_data)
            mps_tensor = qt.Tensor(data=mps_data, inds=("0_reduced", "out"), tags={"0_MPS"})
            tensors.append(mps_tensor)
        else:
            # Create MPO tensors
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
                data = data / torch.norm(data)
                tensors.append(qt.Tensor(data=data, inds=inds, tags=tags))

            # Create MPS tensors
            for i in range(L):
                if i == 0:
                    shape = (reduced_dim, bond_dim)
                    inds = (f"{i}_reduced", f"b_mps_{i}")
                    tags = {f"{i}_MPS"}
                elif i == L - 1:
                    shape = (bond_dim, reduced_dim)
                    inds = (f"b_mps_{i - 1}", f"{i}_reduced")
                    tags = {f"{i}_MPS"}
                else:
                    shape = (bond_dim, reduced_dim, bond_dim)
                    inds = (f"b_mps_{i - 1}", f"{i}_reduced", f"b_mps_{i}")
                    tags = {f"{i}_MPS"}

                if i == self.output_site:
                    shape = shape + (output_dim,)
                    inds = inds + ("out",)

                data = torch.randn(*shape) * base_init
                data = data / torch.norm(data)
                tensors.append(qt.Tensor(data=data, inds=inds, tags=tags))

        self.tn = qt.TensorNetwork(tensors)
        self.input_labels = [f"{i}_in" for i in range(L)]
        self.input_dims = [f"{i}_in" for i in range(L)]
        self.output_dims = ["out"]


class MMPO2:
    """
    Masking MPO2: Non-trainable MPO mask, then trainable MPS.

    Structure:
    - MPO layer: input_dims × input_dims (NOT trainable, cumulative sum mask)
    - MPS layer: input_dims → output_dims (trainable)
    - Tags: {i}_Mask for MPO nodes (with NT tag), {i}_MPS for MPS nodes
    
    Quimb optimal contraction order (L=3):
      0_Mask [0_in, 0_masked, b_mask_0]
        ⊗ 0_MPS [0_masked, b_mps_0]
        contract [0_masked]
        → i0 [0_in, b_mask_0, b_mps_0]
      2_Mask [2_in, 2_masked, b_mask_1]
        ⊗ 2_MPS [2_masked, b_mps_1, out]
        contract [2_masked]
        → i1 [2_in, b_mask_1, b_mps_1, out]
      i0 [0_in, b_mask_0, b_mps_0]
        ⊗ 1_MPS [1_masked, b_mps_0, b_mps_1]
        contract [b_mps_0]
        → i2 [0_in, 1_masked, b_mask_0, b_mps_1]
      i2 [0_in, 1_masked, b_mask_0, b_mps_1]
        ⊗ 1_Mask [1_in, 1_masked, b_mask_0, b_mask_1]
        contract [1_masked, b_mask_0]
        → i3 [0_in, 1_in, b_mask_1, b_mps_1]
      i3 [0_in, 1_in, b_mask_1, b_mps_1]
        ⊗ i1 [2_in, b_mask_1, b_mps_1, out]
        contract [b_mask_1, b_mps_1]
        → i4 [0_in, 1_in, 2_in, out]
      i4 [0_in, 1_in, 2_in, out]
        ⊗ I0 [0_in, s]
        contract [0_in]
        → i5 [1_in, 2_in, out, s]
      i5 [1_in, 2_in, out, s]
        ⊗ I1 [1_in, s]
        contract [1_in]
        → i6 [2_in, out, s]
      i6 [2_in, out, s]
        ⊗ I2 [2_in, s]
        contract [2_in]
        → result [out, s]

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

        tensors = []

        # Special case for L=1: no bond dimensions needed
        if L == 1:
            mask_data = torch.eye(phys_dim)
            mask_tensor = qt.Tensor(data=mask_data, inds=("0_in", "0_masked"), tags={"0_Mask", "NT"})
            tensors.append(mask_tensor)

            mps_data = torch.randn(phys_dim, output_dim) * base_init
            mps_data = mps_data / torch.norm(mps_data)
            mps_tensor = qt.Tensor(data=mps_data, inds=("0_masked", "out"), tags={"0_MPS"})
            tensors.append(mps_tensor) 
        else:
            H = torch.zeros(phys_dim, phys_dim)
            for i in range(phys_dim):
                for j in range(phys_dim):
                    H[i, j] = 1.0 if j >= i else 0.0

            mask_bond_dim = phys_dim

            # Create mask tensors
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
                
                tensors.append(qt.Tensor(data=data, inds=inds, tags=tags))

            # Create MPS tensors
            for i in range(L):
                if i == 0:
                    shape = (phys_dim, bond_dim)
                    inds = (f"{i}_masked", f"b_mps_{i}")
                    tags = {f"{i}_MPS"}
                elif i == L - 1:
                    shape = (bond_dim, phys_dim)
                    inds = (f"b_mps_{i - 1}", f"{i}_masked")
                    tags = {f"{i}_MPS"}
                else:
                    shape = (bond_dim, phys_dim, bond_dim)
                    inds = (f"b_mps_{i - 1}", f"{i}_masked", f"b_mps_{i}")
                    tags = {f"{i}_MPS"}

                if i == self.output_site:
                    shape = shape + (output_dim,)
                    inds = inds + ("out",)

                data = torch.randn(*shape) * base_init
                data = data / torch.norm(data)
                tensors.append(qt.Tensor(data=data, inds=inds, tags=tags))

        self.tn = qt.TensorNetwork(tensors)
        self.input_labels = [f"{i}_in" for i in range(L)]
        self.input_dims = [f"{i}_in" for i in range(L)]
        self.output_dims = ["out"]
