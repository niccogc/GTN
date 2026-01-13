# type: ignore
"""
NTN wrapper for Hierarchical MPS model
"""

from model.NTN import NTN
from model.builder import Inputs
from testing.hierarchical_mps import HierarchicalMPS
import quimb.tensor as qt
from typing import List
import torch


def create_hierarchical_inputs(
    X, y, hierarchical_model: HierarchicalMPS, batch_size: int = 32, batch_dim: str = "s"
):
    """
    Create Inputs object for hierarchical MPS structure.

    Args:
        X: Input data of shape (n_samples, n_patches, pixels_per_patch)
        y: Output data
        hierarchical_model: The hierarchical MPS model instance
        batch_size: Batch size for data loader
        batch_dim: Name for batch dimension

    Returns:
        Inputs object compatible with NTN
    """
    n_samples = X.shape[0]
    n_patches = hierarchical_model.n_patches
    pixels_per_patch = hierarchical_model.pixels_per_patch

    flattened_X = X.reshape(n_samples, n_patches * pixels_per_patch)

    class HierarchicalInputs(Inputs):
        def __init__(self, X_flat, y, input_labels, output_labels, batch_dim, batch_size, h_model):
            self.h_model = h_model
            super().__init__(
                inputs=[X_flat],
                outputs=[y],
                outputs_labels=output_labels,
                input_labels=input_labels,
                batch_dim=batch_dim,
                batch_size=batch_size,
            )

        def _prepare_batch(self, input_data):
            """Override to create hierarchical input structure."""
            flat_data = input_data[0]
            batch_size_actual = flat_data.shape[0]

            reshaped = flat_data.reshape(
                batch_size_actual, self.h_model.n_patches, self.h_model.pixels_per_patch
            )

            return self.h_model.construct_input_nodes(reshaped, batch_dim=self.batch_dim)

    return HierarchicalInputs(
        X_flat=flattened_X,
        y=y,
        input_labels=hierarchical_model.input_labels,
        output_labels=hierarchical_model.output_dims,
        batch_dim=batch_dim,
        batch_size=batch_size,
        h_model=hierarchical_model,
    )
