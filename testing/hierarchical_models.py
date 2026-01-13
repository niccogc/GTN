# type: ignore
"""
Hierarchical MPS models - GCMPO2 for GTN.

Two-level structure:
- Pixel level MPS: one node per site, contracts with pixel dimension
- Patch level MPS: one node per site, contracts with patch dimension

Total nodes: 2 * n_sites
Input: (batch, patches, pixels)
"""

import torch
import quimb.tensor as qt
from typing import Optional
from model.GTN import GTN


def build_hierarchical_mps(
    n_sites: int,
    pixel_dim: int,
    patch_dim: int,
    rank_pixel: int,
    rank_patch: int,
    output_dim: int,
    output_site: Optional[int] = None,
    init_strength: float = 0.01,
):
    """
    Build two-level hierarchical MPS.

    For each site i (0 to n_sites-1):
    - 1 pixel-level node: shape depends on position, contracts with pixels
    - 1 patch-level node: shape depends on position, contracts with patches

    Total: 2 * n_sites nodes
    """
    if output_site is None:
        output_site = n_sites - 1

    tensors = []

    for i in range(n_sites):
        if i == 0:
            pixel_shape = (pixel_dim, rank_pixel)
            pixel_inds = (f"pixel_{i}", f"bp_{i}")
        elif i == n_sites - 1:
            pixel_shape = (rank_pixel, pixel_dim)
            pixel_inds = (f"bp_{i - 1}", f"pixel_{i}")
        else:
            pixel_shape = (rank_pixel, pixel_dim, rank_pixel)
            pixel_inds = (f"bp_{i - 1}", f"pixel_{i}", f"bp_{i}")

        pixel_node = qt.Tensor(
            torch.randn(*pixel_shape) * init_strength, inds=pixel_inds, tags={f"PixelNode_{i}"}
        )
        tensors.append(pixel_node)

    for i in range(n_sites):
        if i == 0:
            patch_shape = (patch_dim, rank_patch)
            patch_inds = (f"patch_{i}", f"bpa_{i}")
        elif i == n_sites - 1:
            patch_shape = (rank_patch, patch_dim)
            patch_inds = (f"bpa_{i - 1}", f"patch_{i}")
        else:
            patch_shape = (rank_patch, patch_dim, rank_patch)
            patch_inds = (f"bpa_{i - 1}", f"patch_{i}", f"bpa_{i}")

        if i == output_site:
            patch_shape = patch_shape + (output_dim,)
            patch_inds = patch_inds + ("out",)

        patch_node = qt.Tensor(
            torch.randn(*patch_shape) * init_strength, inds=patch_inds, tags={f"PatchNode_{i}"}
        )
        tensors.append(patch_node)

    tn = qt.TensorNetwork(tensors)

    input_dims = [f"site_{i}" for i in range(n_sites)]
    output_dims = ["out"]

    return tn, input_dims, output_dims


class GCMPO2(GTN):
    """
    GTN for hierarchical Cross-MPO structure (GCMPO2).

    Input x is already preprocessed as (batch, patch_dim, pixel_dim).
    Each site gets one input node with indices for patches and pixels.
    """

    def construct_nodes(self, x):
        """
        x: tensor shaped as (batch, patch_dim, pixel_dim)

        Creates input nodes - one per site, each contracts with both patch and pixel dimensions.
        """
        input_nodes = []
        for label in self.input_dims:
            i = int(label.split("_")[1])
            node = qt.Tensor(x, inds=["s", f"patch_{i}", f"pixel_{i}"], tags=f"Input_{label}")
            input_nodes.append(node)
        return input_nodes
