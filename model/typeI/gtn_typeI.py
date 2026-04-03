# type: ignore
"""
GTN Type I: Ensemble of GTN models with varying number of sites.

For GTN, the ensemble is simpler than NTN because PyTorch autograd
handles the gradient computation automatically when we sum the outputs.

NOTE: These classes reuse the standard model classes (MPO2, LMPO2, MMPO2, CPDA) from model.standard
to avoid code duplication. The standard classes handle the tensor network construction.
"""

import torch.nn as nn
import quimb.tensor as qt
from typing import Optional

from model.base.GTN import GTN

# Reuse standard model classes instead of duplicating constructor logic
from model.standard.MPO2_models import MPO2, LMPO2, MMPO2
from model.standard.CPD import CPDA


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
        self.phys_dim = phys_dim  # phys_dim includes bias
        self.models = nn.ModuleList()

        for L in range(1, max_sites + 1):
            # First model (L=1) has bias, subsequent models don't
            current_phys_dim = phys_dim if L == 1 else phys_dim - 1
            
            # Use standard MPO2 class instead of duplicate create_simple_mps_tn function
            std_model = MPO2(
                L=L,
                bond_dim=bond_dim,
                phys_dim=current_phys_dim,
                output_dim=output_dim,
                output_site=output_site,
                init_strength=init_strength,
                use_tn_normalization=False,  # TypeI doesn't use normalization
            )
            
            model = GTN_TypeI_Model(
                input_labels=std_model.input_labels,
                tn=std_model.tn,
                output_dims=std_model.output_dims,
                input_dims=std_model.input_dims,
            )
            self.models.append(model)

    def forward(self, x):
        # x has shape (batch, phys_dim) where phys_dim includes bias
        # First model gets full x (with bias), others get x without bias column
        x_no_bias = x[:, :-1]  # Remove last column (bias)

        total = None
        for idx, model in enumerate(self.models):
            x_input = x if idx == 0 else x_no_bias
            y = model(x_input)
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
        self.phys_dim = phys_dim  # phys_dim includes bias
        self.models = nn.ModuleList()

        for L in range(1, max_sites + 1):
            # First model (L=1) has bias, subsequent models don't
            current_phys_dim = phys_dim if L == 1 else phys_dim - 1
            
            # Use standard LMPO2 class instead of duplicate create_lmpo2_tn function
            std_model = LMPO2(
                L=L,
                bond_dim=bond_dim,
                phys_dim=current_phys_dim,
                reduced_dim=reduced_dim,
                output_dim=output_dim,
                output_site=output_site,
                init_strength=init_strength,
                bond_dim_mpo=bond_dim,  # TypeI uses same bond_dim for MPO and MPS
                use_tn_normalization=False,  # TypeI doesn't use normalization
            )
            
            model = GTN_TypeI_Model(
                input_labels=std_model.input_labels,
                tn=std_model.tn,
                output_dims=std_model.output_dims,
                input_dims=std_model.input_dims,
            )
            self.models.append(model)

    def forward(self, x):
        # x has shape (batch, phys_dim) where phys_dim includes bias
        # First model gets full x (with bias), others get x without bias column
        x_no_bias = x[:, :-1]  # Remove last column (bias)

        total = None
        for idx, model in enumerate(self.models):
            x_input = x if idx == 0 else x_no_bias
            y = model(x_input)
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
        self.phys_dim = phys_dim  # phys_dim includes bias
        self.models = nn.ModuleList()

        for L in range(1, max_sites + 1):
            # First model (L=1) has bias, subsequent models don't
            current_phys_dim = phys_dim if L == 1 else phys_dim - 1
            
            # Use standard MMPO2 class instead of duplicate create_mmpo2_tn function
            std_model = MMPO2(
                L=L,
                bond_dim=bond_dim,
                phys_dim=current_phys_dim,
                output_dim=output_dim,
                output_site=output_site,
                init_strength=init_strength,
                use_tn_normalization=False,  # TypeI doesn't use normalization
            )
            
            model = GTN_TypeI_Model(
                input_labels=std_model.input_labels,
                tn=std_model.tn,
                output_dims=std_model.output_dims,
                input_dims=std_model.input_dims,
                not_trainable_tags=["NT"],
            )
            self.models.append(model)

    def forward(self, x):
        x_no_bias = x[:, :-1]  # Remove last column (bias)

        total = None
        for idx, model in enumerate(self.models):
            x_input = x if idx == 0 else x_no_bias
            y = model(x_input)
            if total is None:
                total = y
            else:
                total = total + y
        return total


class CPDATypeI_GTN(nn.Module):
    """Type I ensemble of CPDA models using GTN (autograd)."""

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
        self.phys_dim = phys_dim  # phys_dim includes bias
        self.models = nn.ModuleList()

        for L in range(1, max_sites + 1):
            # First model (L=1) has bias, subsequent models don't
            current_phys_dim = phys_dim if L == 1 else phys_dim - 1
            
            # Use standard CPDA class
            std_model = CPDA(
                L=L,
                bond_dim=bond_dim,
                phys_dim=current_phys_dim,
                output_dim=output_dim,
                output_site=output_site,
                init_strength=init_strength,
                use_tn_normalization=False,  # TypeI doesn't use normalization
            )
            
            model = GTN_TypeI_Model(
                input_labels=std_model.input_labels,
                tn=std_model.tn,
                output_dims=std_model.output_dims,
                input_dims=std_model.input_dims,
            )
            self.models.append(model)

    def forward(self, x):
        # x has shape (batch, phys_dim) where phys_dim includes bias
        # First model gets full x (with bias), others get x without bias column
        x_no_bias = x[:, :-1]  # Remove last column (bias)

        total = None
        for idx, model in enumerate(self.models):
            x_input = x if idx == 0 else x_no_bias
            y = model(x_input)
            if total is None:
                total = y
            else:
                total = total + y
        return total
