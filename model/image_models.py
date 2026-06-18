# type: ignore
import torch
import torch.nn as nn
import quimb.tensor as qt


class CMPO2:
    """
    Convolutional-like MPO2: two MPS chains (pixel + patch) for image patches.
    
    Quimb optimal contraction order (L=3, bond names abstracted):
      0_Pi [0_pixels, bp_0]
        ⊗ I0 [0_patches, 0_pixels, s]
        contract [0_pixels]
        → i0 [0_patches, bp_0, s]
      1_Pi [1_pixels, bp_0, bp_1]
        ⊗ I1 [1_patches, 1_pixels, s]
        contract [1_pixels]
        → i1 [1_patches, bp_0, bp_1, s]
      i0 [0_patches, bp_0, s]
        ⊗ 0_Pa [0_patches, ba_0]
        contract [0_patches]
        → i2 [bp_0, ba_0, s]
      i2 [bp_0, ba_0, s]
        ⊗ i1 [1_patches, bp_0, bp_1, s]
        contract [bp_0]
        → i3 [1_patches, bp_1, ba_0, s]
      i3 [1_patches, bp_1, ba_0, s]
        ⊗ 1_Pa [1_patches, ba_0, ba_1]
        contract [1_patches, ba_0]
        → i4 [bp_1, ba_1, s]
      i4 [bp_1, ba_1, s]
        ⊗ 2_Pa [2_patches, ba_1]
        contract [ba_1]
        → i5 [2_patches, bp_1, s]
      i5 [2_patches, bp_1, s]
        ⊗ I2 [2_patches, 2_pixels, s]
        contract [2_patches]
        → i6 [2_pixels, bp_1, s]
      i6 [2_pixels, bp_1, s]
        ⊗ 2_Pi [2_pixels, bp_1, out]
        contract [2_pixels, bp_1]
        → result [out, s]
    
    bp: pixel bond dims | ba: patch bond dims
    """
    def __init__(
        self,
        L: int,
        pixel_dim: int,
        patch_dim: int,
        pixel_bond_dim: int,
        patch_bond_dim: int,
        output_dim: int,
        init_strength: float = 0.01,
    ):
        self.L = L
        self.pixel_dim = pixel_dim
        self.patch_dim = patch_dim
        self.output_dim = output_dim

        psi = qt.MPS_rand_state(L, bond_dim=pixel_bond_dim, phys_dim=pixel_dim)
        phi = qt.MPS_rand_state(L, bond_dim=patch_bond_dim, phys_dim=patch_dim)

        output_node = psi[f"I{L - 1}"]
        output_node.new_ind(
            "out", size=output_dim, axis=-1, mode="random", rand_strength=init_strength
        )

        psi.apply_to_arrays(lambda x: torch.tensor(x, dtype=torch.float64))
        phi.apply_to_arrays(lambda x: torch.tensor(x, dtype=torch.float64))

        psi.reindex({f"k{i}": f"{i}_pixels" for i in range(L)}, inplace=True)
        phi.reindex({f"k{i}": f"{i}_patches" for i in range(L)}, inplace=True)

        for i in range(L):
            psi_tensor = psi[f"I{i}"]
            psi_tensor.drop_tags(f"I{i}")
            psi_tensor.add_tag(f"{i}_Pi")

            phi_tensor = phi[f"I{i}"]
            phi_tensor.drop_tags(f"I{i}")
            phi_tensor.add_tag(f"{i}_Pa")

        self.tn = psi & phi
        self.input_labels = [[0, (f"{i}_patches", f"{i}_pixels")] for i in range(L)]
        self.input_dims = [str(i) for i in range(L)]
        self.output_dims = ["out"]


class CMPO3:
    def __init__(
        self,
        L: int,
        channel_dim: int,
        pixel_dim: int,
        patch_dim: int,
        channel_bond_dim: int,
        pixel_bond_dim: int,
        patch_bond_dim: int,
        output_dim: int,
        init_strength: float = 0.01,
    ):
        self.L = L
        self.channel_dim = channel_dim
        self.pixel_dim = pixel_dim
        self.patch_dim = patch_dim
        self.output_dim = output_dim

        chi = qt.MPS_rand_state(L, bond_dim=channel_bond_dim, phys_dim=channel_dim)
        psi = qt.MPS_rand_state(L, bond_dim=pixel_bond_dim, phys_dim=pixel_dim)
        phi = qt.MPS_rand_state(L, bond_dim=patch_bond_dim, phys_dim=patch_dim)

        output_node = chi[f"I{L - 1}"]
        output_node.new_ind(
            "out", size=output_dim, axis=-1, mode="random", rand_strength=init_strength
        )

        chi.apply_to_arrays(lambda x: torch.tensor(x, dtype=torch.float64))
        psi.apply_to_arrays(lambda x: torch.tensor(x, dtype=torch.float64))
        phi.apply_to_arrays(lambda x: torch.tensor(x, dtype=torch.float64))

        chi.reindex({f"k{i}": f"{i}_channels" for i in range(L)}, inplace=True)
        psi.reindex({f"k{i}": f"{i}_pixels" for i in range(L)}, inplace=True)
        phi.reindex({f"k{i}": f"{i}_patches" for i in range(L)}, inplace=True)

        for i in range(L):
            chi_tensor = chi[f"I{i}"]
            chi_tensor.drop_tags(f"I{i}")
            chi_tensor.add_tag(f"{i}_Ch")

            psi_tensor = psi[f"I{i}"]
            psi_tensor.drop_tags(f"I{i}")
            psi_tensor.add_tag(f"{i}_Pi")

            phi_tensor = phi[f"I{i}"]
            phi_tensor.drop_tags(f"I{i}")
            phi_tensor.add_tag(f"{i}_Pa")

        self.tn = chi & psi & phi
        self.input_labels = [
            [0, (f"{i}_patches", f"{i}_pixels", f"{i}_channels")] for i in range(L)
        ]
        self.input_dims = [str(i) for i in range(L)]
        self.output_dims = ["out"]

# The CRING models follow the idea of CMPO2 where the networks are indipendently attached to the different inputs dimensions.  BUT YOU ONLY NEED ONE input. since check the BosonMPS, we contract the input, the power, then the trace. So in the following after we contract all input dimensions, we are left with left ranks labels, right ranks labels, to be merged into one using quimb To obtain a matrix, then we do the power of that matrix.
# 
# TODO: The C2Ring_GTN, is just two BosonMPS, as defined in /home/nicci/Desktop/remote/GTN/model/standard/BosonMPS.py, one for the patches and one for the pixels
# class CRing_GTN(nn.Module):
    # return

# TODO: The C3Ring_GTN, is just two BosonMPS, as defined in /home/nicci/Desktop/remote/GTN/model/standard/BosonMPS.py, one for the patches and one for the pixels and one for the channels
# class C3Ring_GTN(nn.Module):
    # return

class BaselineCNN(nn.Module):
    """
    Baseline CNN for comparison with tensor network models.
    Architecture: Conv blocks (with pooling) -> Flatten -> FC layers -> Output
    """
    def __init__(
        self,
        input_channels: int = 1,
        image_size: int = 28,
        n_classes: int = 10,
        n_conv_layers: int = 2,
        base_channels: int = 16,
        fc_hidden_dim: int = 128,
        kernel_size: int = 3,
        use_batchnorm: bool = True,
    ):
        super().__init__()
        self.n_conv_layers = n_conv_layers
        self.base_channels = base_channels
        
        conv_layers = []
        in_ch = input_channels
        out_ch = base_channels
        spatial_size = image_size
        
        for i in range(n_conv_layers):
            conv_layers.append(nn.Conv2d(in_ch, out_ch, kernel_size, padding=kernel_size//2))
            if use_batchnorm:
                conv_layers.append(nn.BatchNorm2d(out_ch))
            conv_layers.append(nn.ReLU(inplace=True))
            conv_layers.append(nn.MaxPool2d(2, 2))
            spatial_size = spatial_size // 2
            in_ch = out_ch
            out_ch = min(out_ch * 2, 256)
        
        self.features = nn.Sequential(*conv_layers)
        
        flatten_dim = in_ch * spatial_size * spatial_size
        
        if fc_hidden_dim > 0:
            self.classifier = nn.Sequential(
                nn.Linear(flatten_dim, fc_hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(fc_hidden_dim, n_classes),
            )
        else:
            self.classifier = nn.Linear(flatten_dim, n_classes)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class CMPO2_GTN(nn.Module):
    def __init__(self, cmpo2: CMPO2):
        super().__init__()
        self.L = cmpo2.L
        self.input_labels = cmpo2.input_labels
        self.output_dims = cmpo2.output_dims

        params, self.skeleton = qt.pack(cmpo2.tn)
        self.torch_params = nn.ParameterDict(
            {str(i): nn.Parameter(p) for i, p in params.items()}
        )

    def forward(self, x):
        batch_size, n_patches, pixels_per_patch = x.shape

        tn_params = {int(i): p for i, p in self.torch_params.items()}
        weights = qt.unpack(tn_params, self.skeleton)
        
        input_nodes = []
        for i in range(self.L):
            _, (patch_idx, pixel_idx) = self.input_labels[i]
            input_node = qt.Tensor(x, inds=["s", patch_idx, pixel_idx], tags=f"Input_{i}")
            input_nodes.append(input_node)

        input_tn = qt.TensorNetwork(input_nodes)
        full_tn = weights & input_tn
        out = full_tn.contract(output_inds=["s"] + self.output_dims, optimize="auto-hq")
        return out.data


class CMPO3_GTN(nn.Module):
    def __init__(self, cmpo3: CMPO3):
        super().__init__()
        self.L = cmpo3.L
        self.input_labels = cmpo3.input_labels
        self.output_dims = cmpo3.output_dims

        params, self.skeleton = qt.pack(cmpo3.tn)
        self.torch_params = nn.ParameterDict(
            {str(i): nn.Parameter(p) for i, p in params.items()}
        )

    def forward(self, x):
        batch_size, n_patches, pixels_per_patch, n_channels = x.shape

        tn_params = {int(i): p for i, p in self.torch_params.items()}
        weights = qt.unpack(tn_params, self.skeleton)

        input_nodes = []
        for i in range(self.L):
            _, (patch_idx, pixel_idx, channel_idx) = self.input_labels[i]
            input_node = qt.Tensor(x, inds=["s", patch_idx, pixel_idx, channel_idx], tags=f"Input_{i}")
            input_nodes.append(input_node)

        input_tn = qt.TensorNetwork(input_nodes)
        full_tn = weights & input_tn
        out = full_tn.contract(output_inds=["s"] + self.output_dims, optimize="auto-hq")
        return out.data
