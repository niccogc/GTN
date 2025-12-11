from typing import List, Dict, Optional, Tuple
import torch.nn as nn
import quimb.tensor as qt

NOT_TRAINABLE_TAG = "NT"

class GTN(nn.Module):
    def __init__(self, tn, output_dims, input_dims):
        super().__init__()
        
        params, self.skeleton_pixels = qt.pack(tn)

        self.torch_params= nn.ParameterDict({
            str(i): nn.Parameter(initial) for i, initial in params.items()
        })
        
        self.input_dims = input_dims
        self.output_dims = output_dims

    def forward(self, x):
        # Unpack parameters
        tn_params= {int(i): p for i, p in self.torch_params.items()}
        weights = qt.unpack(tn_params, self.skeleton_pixels)
        
        # Construct Input Nodes
        input_nodes_list = self.construct_nodes(x)
        input_tn = qt.TensorNetwork(input_nodes_list)

        tn = weights & input_tn
        
        # Contract and return raw data
        out = tn.contract(output_inds=self.output_dims, optimize='auto-hq')
        return out.data


    def construct_nodes(self, x):
        """
            This depends on the model and inut dims. can be defined for each model.
        """
        if len(x) > 1:
            assert len(x) == len(self.input_dims)
            return [qt.Tensor(i, inds = ["s", f"{j}"], tags=f"Input_{j}") for i,j in zip(x, self.input_dims)]
        input_nodes = []
        for i in self.input_dims:
            a = qt.Tensor(x, inds=["s", f"{i}"], tags=f"Input_{i}")
            input_nodes.append(a)
        return input_nodes

