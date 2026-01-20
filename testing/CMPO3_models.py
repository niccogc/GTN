# type: ignore
"""
CMPO3 (Cross MPO3) Models: Three-level tensor network structure.

Structure:
- Channel level MPS: processes color channels
- Pixel level MPS: processes pixels within patches
- Patch level MPS: processes patch-level features
- Input format: (batch, n_patches, pixels_per_patch, n_channels)
"""

import torch
import quimb.tensor as qt
from typing import Optional
from model.base.GTN import GTN

torch.set_default_dtype(torch.float64)


def _create_cmpo3(
    n_sites: int,
    channel_dim: int,
    pixel_dim: int,
    patch_dim: int,
    rank_channel: int,
    rank_pixel: int,
    rank_patch: int,
    output_dim: int,
    output_site: Optional[int] = None,
    init_strength: float = 0.01,
):
    if output_site is None:
        output_site = n_sites - 1

    tensors = []

    def make_mps_nodes(n_sites, phys_dim, rank, prefix, tag_prefix):
        nodes = []
        for i in range(n_sites):
            if n_sites == 1:
                shape = (phys_dim,)
                inds = (f"{prefix}_{i}",)
            elif i == 0:
                shape = (phys_dim, rank)
                inds = (f"{prefix}_{i}", f"b{prefix}_{i}")
            elif i == n_sites - 1:
                shape = (rank, phys_dim)
                inds = (f"b{prefix}_{i - 1}", f"{prefix}_{i}")
            else:
                shape = (rank, phys_dim, rank)
                inds = (f"b{prefix}_{i - 1}", f"{prefix}_{i}", f"b{prefix}_{i}")

            node = qt.Tensor(
                torch.randn(*shape) * init_strength, inds=inds, tags={f"{tag_prefix}_{i}"}
            )
            nodes.append(node)
        return nodes

    channel_nodes = make_mps_nodes(n_sites, channel_dim, rank_channel, "ch", "ChannelNode")
    pixel_nodes = make_mps_nodes(n_sites, pixel_dim, rank_pixel, "px", "PixelNode")

    for i in range(n_sites):
        if n_sites == 1:
            shape = (patch_dim,)
            inds = (f"pa_{i}",)
        elif i == 0:
            shape = (patch_dim, rank_patch)
            inds = (f"pa_{i}", f"bpa_{i}")
        elif i == n_sites - 1:
            shape = (rank_patch, patch_dim)
            inds = (f"bpa_{i - 1}", f"pa_{i}")
        else:
            shape = (rank_patch, patch_dim, rank_patch)
            inds = (f"bpa_{i - 1}", f"pa_{i}", f"bpa_{i}")

        if i == output_site:
            shape = shape + (output_dim,)
            inds = inds + ("out",)

        patch_node = qt.Tensor(
            torch.randn(*shape) * init_strength, inds=inds, tags={f"PatchNode_{i}"}
        )
        tensors.append(patch_node)

    tensors.extend(channel_nodes)
    tensors.extend(pixel_nodes)

    tn = qt.TensorNetwork(tensors)

    input_labels = [(f"pa_{i}", f"px_{i}", f"ch_{i}") for i in range(n_sites)]
    input_dims = [f"site_{i}" for i in range(n_sites)]
    output_dims = ["out"]

    return tn, input_labels, input_dims, output_dims


class CMPO3:
    def __init__(
        self,
        n_sites: int,
        channel_dim: int,
        pixel_dim: int,
        patch_dim: int,
        rank_channel: int,
        rank_pixel: int,
        rank_patch: int,
        output_dim: int,
        output_site: Optional[int] = None,
        init_strength: float = 0.01,
    ):
        self.n_sites = n_sites
        self.channel_dim = channel_dim
        self.pixel_dim = pixel_dim
        self.patch_dim = patch_dim
        self.rank_channel = rank_channel
        self.rank_pixel = rank_pixel
        self.rank_patch = rank_patch
        self.output_dim = output_dim

        tn, input_labels, input_dims, output_dims = _create_cmpo3(
            n_sites=n_sites,
            channel_dim=channel_dim,
            pixel_dim=pixel_dim,
            patch_dim=patch_dim,
            rank_channel=rank_channel,
            rank_pixel=rank_pixel,
            rank_patch=rank_patch,
            output_dim=output_dim,
            output_site=output_site,
            init_strength=init_strength,
        )

        self.tn = tn
        self.input_labels = input_labels
        self.input_dims = input_dims
        self.output_dims = output_dims


class CMPO3_GTN(GTN):
    def __init__(self, cmpo3: CMPO3):
        self.cmpo3 = cmpo3
        self.n_sites = cmpo3.n_sites

        super().__init__(
            tn=cmpo3.tn,
            output_dims=cmpo3.output_dims,
            input_dims=cmpo3.input_dims,
        )

    def construct_nodes(self, x):
        input_nodes = []
        for i in range(self.n_sites):
            node = qt.Tensor(x, inds=["s", f"pa_{i}", f"px_{i}", f"ch_{i}"], tags=f"Input_site_{i}")
            input_nodes.append(node)
        return input_nodes


if __name__ == "__main__":
    batch, n_patches, pixels_per_patch, n_channels = 4, 8, 32, 3
    x = torch.randn(batch, n_patches, pixels_per_patch, n_channels, dtype=torch.float64)

    cmpo3 = CMPO3(
        n_sites=2,
        channel_dim=n_channels,
        pixel_dim=pixels_per_patch,
        patch_dim=n_patches,
        rank_channel=4,
        rank_pixel=4,
        rank_patch=4,
        output_dim=10,
    )
    model = CMPO3_GTN(cmpo3)
    out = model(x)
    print(f"Input: {x.shape}, Output: {out.shape}")
