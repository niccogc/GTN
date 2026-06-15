# type: ignore
import os
from typing import List, Dict, Optional, Tuple
import torch
import torch.nn as nn
import quimb.tensor as qt
import cotengra as ctg

# =============================================================================
# Quimb Contraction Strategy Configuration
# =============================================================================
# Default: ReusableHyperOptimizer with path caching for fastest contraction execution.
# Paths are cached to disk, so repeated contractions with same structure are instant.
#
# Environment variables:
#   QUIMB_CONTRACT_STRATEGY: Override with a string strategy
#     - "greedy"           : Super fast path-finding, decent paths
#     - "auto"             : quimb default, balanced
#     - "auto-hq"          : Higher quality paths
#     - "random-greedy"    : Good for large networks
#     - "random-greedy-128": Same but with 4x more trials  
#     - "optimal"          : Best paths, exponential path-finding cost
#
#   QUIMB_CONTRACT_MINIMIZE: What to optimize for (default: "flops")
#     - "flops"  : Fastest contraction execution (default)
#     - "write"  : Lowest memory usage
#     - "combo"  : Balance of both

def _get_contract_strategy():
    """Get the contraction strategy from environment or default."""
    # Check for string strategy override
    strategy_override = os.environ.get("QUIMB_CONTRACT_STRATEGY", "")
    
    if strategy_override:
        valid_strategies = {"greedy", "auto", "auto-hq", "random-greedy", "random-greedy-128", "optimal"}
        if strategy_override not in valid_strategies:
            import logging
            logging.getLogger(__name__).warning(
                f"Unknown QUIMB_CONTRACT_STRATEGY='{strategy_override}', using default. "
                f"Valid options: {valid_strategies}"
            )
        else:
            return strategy_override
    
    # Default: ReusableHyperOptimizer with caching for fastest contractions
    minimize = os.environ.get("QUIMB_CONTRACT_MINIMIZE", "flops")
    if minimize not in {"flops", "write", "combo"}:
        import logging
        logging.getLogger(__name__).warning(
            f"Unknown QUIMB_CONTRACT_MINIMIZE='{minimize}', using 'flops'. "
            f"Valid options: flops, write, combo"
        )
        minimize = "flops"
    
    return ctg.ReusableHyperOptimizer(
        minimize=minimize,          # Optimize for fastest contraction by default
        reconf_opts={},             # Enable subtree reconfiguration
        max_time="rate:1e8",        # Only spend time on hard contractions
        hash_method="b",            # Hash up to index permutation for max reuse
        directory=True,             # Cache paths to disk
        progbar=False,
    )

_contract_strategy = _get_contract_strategy()
qt.set_contract_strategy(_contract_strategy)

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

    def _apply(self, fn):
        """Override _apply to also move not_trainable_params."""
        super()._apply(fn)
        self.not_trainable_params = {
            k: fn(v) if isinstance(v, torch.Tensor) else v
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
        Construct input tensor nodes for the tensor network.
        
        Args:
            x: Either a batched tensor (batch, features) or a list of tensors (one per input dim)
        """
        # Check if x is a list/tuple of separate tensors (one per input dimension)
        if isinstance(x, (list, tuple)):
            assert len(x) == len(self.input_dims)
            return [
                qt.Tensor(i, inds=["s", f"{j}"], tags=f"Input_{j}")
                for i, j in zip(x, self.input_dims)
            ]
        
        # x is a single batched tensor - create one input node per dimension
        input_nodes = []
        for i in self.input_dims:
            a = qt.Tensor(x, inds=["s", f"{i}"], tags=f"Input_{i}")
            input_nodes.append(a)
        return input_nodes
