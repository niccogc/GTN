# type: ignore
"""
GTN wrapper for Hierarchical MPS model
"""

from model.GTN import GTN
from testing.hierarchical_mps import HierarchicalMPS
import quimb.tensor as qt


class HierarchicalGTN(GTN):
    """GTN wrapper for hierarchical MPS that handles patch-pixel structured inputs."""

    def __init__(self, hierarchical_mps: HierarchicalMPS):
        self.hierarchical_model = hierarchical_mps

        super().__init__(
            tn=hierarchical_mps.tn,
            output_dims=hierarchical_mps.output_dims,
            input_dims=hierarchical_mps.input_dims,
        )

    def construct_nodes(self, x):
        """
        Construct input nodes for hierarchical structure.

        x should be shape: (batch, n_patches, pixels_per_patch)
        """
        return self.hierarchical_model.construct_input_nodes(x, batch_dim="s")
