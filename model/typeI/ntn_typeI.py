# type: ignore
"""
Type I Model Builders: MPO2TypeI, LMPO2TypeI, MMPO2TypeI, CPDATypeI

These classes construct ensembles of tensor networks with varying number of sites (L=1,2,...,max_sites).
They expose the same interface pattern as standard models but with lists:
- .tns: List of TensorNetworks
- .input_dims_list: List of input_dims for each TN
- .input_labels_list: List of input_labels for each TN
- .output_dims: Output dimensions (same for all TNs)

NOTE: These classes reuse the standard model classes (MPO2, LMPO2, MMPO2, CPDA) from model.standard
to avoid code duplication. The standard classes handle the tensor network construction.
"""
import quimb.tensor as qt
from typing import List, Optional
from model.standard.MPO2_models import MPO2, LMPO2, MMPO2
from model.standard.CPD import CPDA


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
            # First TN (L=1) has bias, subsequent TNs don't
            current_phys_dim = phys_dim if L == 1 else phys_dim - 1
            
            # Use standard MPO2 class instead of duplicate _create_simple_mps function
            model = MPO2(
                L=L,
                bond_dim=bond_dim,
                phys_dim=current_phys_dim,
                output_dim=output_dim,
                output_site=output_site,
                init_strength=init_strength,
                use_tn_normalization=False,  # TypeI doesn't use normalization
            )
            
            self.tns.append(model.tn)
            self.input_dims_list.append(model.input_dims)
            self.input_labels_list.append(model.input_labels)


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
            # First TN (L=1) has bias, subsequent TNs don't
            current_phys_dim = phys_dim if L == 1 else phys_dim - 1
            
            # Use standard LMPO2 class instead of duplicate _create_lmpo2 function
            model = LMPO2(
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
            
            self.tns.append(model.tn)
            self.input_dims_list.append(model.input_dims)
            self.input_labels_list.append(model.input_labels)


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
            # First TN (L=1) has bias, subsequent TNs don't
            current_phys_dim = phys_dim if L == 1 else phys_dim - 1
            
            # Use standard MMPO2 class instead of duplicate _create_mmpo2 function
            model = MMPO2(
                L=L,
                bond_dim=bond_dim,
                phys_dim=current_phys_dim,
                output_dim=output_dim,
                output_site=output_site,
                init_strength=init_strength,
                use_tn_normalization=False,  # TypeI doesn't use normalization
            )
            
            self.tns.append(model.tn)
            self.input_dims_list.append(model.input_dims)
            self.input_labels_list.append(model.input_labels)


class CPDATypeI:
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
            # First TN (L=1) has bias, subsequent TNs don't
            current_phys_dim = phys_dim if L == 1 else phys_dim - 1
            
            # Use standard CPDA class
            model = CPDA(
                L=L,
                bond_dim=bond_dim,
                phys_dim=current_phys_dim,
                output_dim=output_dim,
                output_site=output_site,
                init_strength=init_strength,
                use_tn_normalization=False,  # TypeI doesn't use normalization
            )
            
            self.tns.append(model.tn)
            self.input_dims_list.append(model.input_dims)
            self.input_labels_list.append(model.input_labels)
