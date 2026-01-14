# type: ignore
"""
Hierarchical MPS Model: Two-level structure (patches and pixels)

Structure:
- Pixel level MPS: processes individual pixels within patches
- Patch level MPS: processes patch-level features
- Input format: batch_dim x n_patches x pixels_per_patch
"""

import torch
import quimb.tensor as qt
from typing import Optional, List


class HierarchicalMPS:
    """
    Two-level hierarchical MPS for image classification.

    Level 1 (Pixel): node_pixel_i has shape (pixel_dim, rank_pixel) for first/last,
                     or (rank_pixel, pixel_dim, rank_pixel) for middle nodes

    Level 2 (Patch): node_patch_i has shape (patch_dim, rank_patch) for first/last,
                     or (rank_patch, patch_dim, rank_patch) for middle nodes

    The two levels are connected via shared indices.
    """

    def __init__(
        self,
        n_patches: int,
        pixels_per_patch: int,
        rank_pixel: int,
        rank_patch: int,
        output_dim: int,
        output_site: Optional[int] = None,
        init_strength: float = 0.01,
    ):
        """
        Args:
            n_patches: Number of patches in the image
            pixels_per_patch: Number of pixels in each patch
            rank_pixel: Bond dimension for pixel-level MPS
            rank_patch: Bond dimension for patch-level MPS
            output_dim: Output dimension (e.g., number of classes)
            output_site: Which patch gets the output dimension (default: last patch)
            init_strength: Initialization strength
        """
        self.n_patches = n_patches
        self.pixels_per_patch = pixels_per_patch
        self.rank_pixel = rank_pixel
        self.rank_patch = rank_patch
        self.output_dim = output_dim
        self.output_site = output_site if output_site is not None else n_patches - 1

        # Build pixel-level MPS tensors (one chain per patch)
        pixel_tensors = []
        for patch_idx in range(n_patches):
            for pixel_idx in range(pixels_per_patch):
                if pixel_idx == 0:
                    # First pixel in patch: (pixel_dim, rank_pixel)
                    shape = (1, rank_pixel)  # pixel_dim will come from input contraction
                    inds = (f"pixel_{patch_idx}_{pixel_idx}", f"pb_{patch_idx}_{pixel_idx}")
                elif pixel_idx == pixels_per_patch - 1:
                    # Last pixel in patch: (rank_pixel, pixel_dim) + patch connection
                    shape = (rank_pixel, 1)  # pixel_dim from input, connect to patch level
                    inds = (f"pb_{patch_idx}_{pixel_idx - 1}", f"pixel_{patch_idx}_{pixel_idx}")
                else:
                    # Middle pixels: (rank_pixel, pixel_dim, rank_pixel)
                    shape = (rank_pixel, 1, rank_pixel)
                    inds = (
                        f"pb_{patch_idx}_{pixel_idx - 1}",
                        f"pixel_{patch_idx}_{pixel_idx}",
                        f"pb_{patch_idx}_{pixel_idx}",
                    )

                # For last pixel in patch, add connection to patch level
                if pixel_idx == pixels_per_patch - 1:
                    shape = shape + (rank_patch,)  # Add patch-level connection
                    inds = inds + (f"patch_connect_{patch_idx}",)

                data = torch.randn(*shape) * init_strength
                tensor = qt.Tensor(
                    data=data, inds=inds, tags={f"PixelNode_{patch_idx}_{pixel_idx}"}
                )
                pixel_tensors.append(tensor)

        # Build patch-level MPS tensors
        patch_tensors = []
        for patch_idx in range(n_patches):
            if patch_idx == 0:
                # First patch: (patch_dim, rank_patch)
                shape = (rank_patch, rank_patch)  # Input from pixel level
                inds = (f"patch_connect_{patch_idx}", f"b_patch_{patch_idx}")
            elif patch_idx == n_patches - 1:
                # Last patch: (rank_patch, patch_dim)
                shape = (rank_patch, rank_patch)
                inds = (f"b_patch_{patch_idx - 1}", f"patch_connect_{patch_idx}")
            else:
                # Middle patches: (rank_patch, patch_dim, rank_patch)
                shape = (rank_patch, rank_patch, rank_patch)
                inds = (
                    f"b_patch_{patch_idx - 1}",
                    f"patch_connect_{patch_idx}",
                    f"b_patch_{patch_idx}",
                )

            # Add output dimension to specified patch
            if patch_idx == self.output_site:
                shape = shape + (output_dim,)
                inds = inds + ("out",)

            data = torch.randn(*shape) * init_strength
            tensor = qt.Tensor(data=data, inds=inds, tags={f"PatchNode_{patch_idx}"})
            patch_tensors.append(tensor)

        # Combine into full tensor network
        self.tn = qt.TensorNetwork(pixel_tensors + patch_tensors)

        # Define input structure for NTN/GTN
        # Each site corresponds to a (patch, pixel) pair
        self.input_labels = []
        for patch_idx in range(n_patches):
            for pixel_idx in range(pixels_per_patch):
                self.input_labels.append(f"pixel_{patch_idx}_{pixel_idx}")

        self.input_dims = self.input_labels.copy()
        self.output_dims = ["out"]

    def construct_input_nodes(self, x: torch.Tensor, batch_dim: str = "s") -> List[qt.Tensor]:
        """
        Construct input tensor nodes from image batch.

        Args:
            x: Input tensor of shape (batch, n_patches, pixels_per_patch)
            batch_dim: Name for batch dimension

        Returns:
            List of quimb Tensor nodes, one per (patch, pixel) location
        """
        batch_size = x.shape[0]
        input_nodes = []

        for patch_idx in range(self.n_patches):
            for pixel_idx in range(self.pixels_per_patch):
                # Extract pixel data: shape (batch_size, 1)
                pixel_data = x[:, patch_idx, pixel_idx].unsqueeze(-1)

                # Create input tensor node
                label = f"pixel_{patch_idx}_{pixel_idx}"
                node = qt.Tensor(
                    data=pixel_data,
                    inds=(batch_dim, label),
                    tags={f"Input_{patch_idx}_{pixel_idx}"},
                )
                input_nodes.append(node)

        return input_nodes
