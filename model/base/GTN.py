# type: ignore
from typing import List, Dict, Optional, Tuple
import torch.nn as nn
import quimb.tensor as qt

NOT_TRAINABLE_TAG = "NT"


class GTN(nn.Module):
    def __init__(self, tn, output_dims, input_dims, not_trainable_tags=None):
        super().__init__()

        not_trainable_tags = not_trainable_tags or []
        all_tags_to_exclude = set(not_trainable_tags)
        all_tags_to_exclude.add(NOT_TRAINABLE_TAG)

        trainable_mask = {}
        for idx, tensor in enumerate(tn.tensors):
            is_trainable = not any(tag in all_tags_to_exclude for tag in tensor.tags)
            trainable_mask[idx] = is_trainable

        params, self.skeleton_pixels = qt.pack(tn)

        self.torch_params = nn.ParameterDict(
            {
                str(i): nn.Parameter(initial)
                for i, initial in params.items()
                if trainable_mask.get(i, True)  # Default to trainable if not found
            }
        )

        self.not_trainable_params = {
            i: initial.clone() for i, initial in params.items() if not trainable_mask.get(i, True)
        }

        self.input_dims = input_dims
        self.output_dims = output_dims

    def to(self, *args, **kwargs):
        # Move the module to device
        super().to(*args, **kwargs)
        # Also move non-trainable parameters
        device = args[0] if args else kwargs.get("device", None)
        if device is not None:
            self.not_trainable_params = {
                k: v.to(device) if hasattr(v, "to") else v
                for k, v in self.not_trainable_params.items()
            }
        return self

    def forward(self, x):
        tn_params = {int(i): p for i, p in self.torch_params.items()}
        tn_params.update(self.not_trainable_params)  # Add frozen params
        weights = qt.unpack(tn_params, self.skeleton_pixels)

        input_nodes_list = self.construct_nodes(x)
        input_tn = qt.TensorNetwork(input_nodes_list)

        tn = weights & input_tn

        out = tn.contract(output_inds=["s"] + self.output_dims, optimize="auto-hq")
        return out.data

    def construct_nodes(self, x):
        """
        This depends on the model and input dims. can be defined for each model.
        """
        if len(x) > 1:
            assert len(x) == len(self.input_dims)
            return [
                qt.Tensor(i, inds=["s", f"{j}"], tags=f"Input_{j}")
                for i, j in zip(x, self.input_dims)
            ]
        input_nodes = []
        for i in self.input_dims:
            a = qt.Tensor(x, inds=["s", f"{i}"], tags=f"Input_{i}")
            input_nodes.append(a)
        return input_nodes
