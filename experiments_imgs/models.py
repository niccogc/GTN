# type: ignore
import torch
import torch.nn as nn
import quimb.tensor as qt


class CMPO2:
    def __init__(
        self,
        L: int,
        pixel_dim: int,
        patch_dim: int,
        bond_dim: int,
        output_dim: int,
        init_strength: float = 0.01,
    ):
        self.L = L
        self.pixel_dim = pixel_dim
        self.patch_dim = patch_dim
        self.output_dim = output_dim

        psi = qt.MPS_rand_state(L, bond_dim=bond_dim, phys_dim=pixel_dim)
        phi = qt.MPS_rand_state(L, bond_dim=bond_dim, phys_dim=patch_dim)

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
        rank_channel: int,
        bond_dim: int,
        output_dim: int,
        init_strength: float = 0.01,
    ):
        self.L = L
        self.channel_dim = channel_dim
        self.pixel_dim = pixel_dim
        self.patch_dim = patch_dim
        self.output_dim = output_dim

        chi = qt.MPS_rand_state(L, bond_dim=rank_channel, phys_dim=channel_dim)
        psi = qt.MPS_rand_state(L, bond_dim=bond_dim, phys_dim=pixel_dim)
        phi = qt.MPS_rand_state(L, bond_dim=bond_dim, phys_dim=patch_dim)

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
            site_idx = i % n_patches

            pixel_data = x[:, site_idx, :]
            t_pixel = qt.Tensor(pixel_data, inds=["s", pixel_idx], tags=f"In_pixel_{i}")
            input_nodes.append(t_pixel)

            patch_one_hot = torch.zeros(
                batch_size, n_patches, dtype=x.dtype, device=x.device
            )
            patch_one_hot[:, site_idx] = 1.0
            t_patch = qt.Tensor(
                patch_one_hot, inds=["s", patch_idx], tags=f"In_patch_{i}"
            )
            input_nodes.append(t_patch)

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
            site_idx = i % n_patches

            data = x[:, site_idx, :, :]

            pixel_data = data.mean(dim=-1)
            t_pixel = qt.Tensor(pixel_data, inds=["s", pixel_idx], tags=f"In_pixel_{i}")
            input_nodes.append(t_pixel)

            channel_data = data.mean(dim=-2)
            t_channel = qt.Tensor(
                channel_data, inds=["s", channel_idx], tags=f"In_channel_{i}"
            )
            input_nodes.append(t_channel)

            patch_one_hot = torch.zeros(
                batch_size, n_patches, dtype=x.dtype, device=x.device
            )
            patch_one_hot[:, site_idx] = 1.0
            t_patch = qt.Tensor(
                patch_one_hot, inds=["s", patch_idx], tags=f"In_patch_{i}"
            )
            input_nodes.append(t_patch)

        input_tn = qt.TensorNetwork(input_nodes)
        full_tn = weights & input_tn
        out = full_tn.contract(output_inds=["s"] + self.output_dims, optimize="auto-hq")
        return out.data
