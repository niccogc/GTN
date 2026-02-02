# type: ignore
"""
PartitionRank Model Classes

PartitionRank models are ensemble models where each term partitions one variable
from the others. For 3 sites, the model is:

f(x0, x1, x2) = A1(x0, ra, out) * A2(ra, x1, x2) +
               B1(x1, rb, out) * B2(rb, x0, x2) +
               C1(x2, rc, out) * C2(rc, x0, x1)

Each term partitions one input variable (the "pivot") from the remaining variables,
connecting them through a rank index.

The "partition rank" is defined as: partition_rank = ra + rb + rc
"""

import torch
import quimb.tensor as qt
from typing import List, Optional, Tuple, Union


def distribute_partition_rank(total_rank: int, n_terms: int = 3) -> Tuple[int, ...]:
    """
    Equidistribute total partition rank across terms.

    Examples for n_terms=3:
        total_rank=3 -> (1, 1, 1)
        total_rank=4 -> (2, 1, 1)
        total_rank=5 -> (2, 2, 1)
        total_rank=6 -> (2, 2, 2)
        total_rank=7 -> (3, 2, 2)
    """
    base = total_rank // n_terms
    remainder = total_rank % n_terms
    ranks = [base] * n_terms
    for i in range(remainder):
        ranks[i] += 1
    return tuple(ranks)


class PartitionRank3:
    """
    PartitionRank model for 3 sites.

    Structure:
        f(x0, x1, x2) = A1(x0, ra, out) * A2(ra, x1, x2) +
                       B1(x1, rb, out) * B2(rb, x0, x2) +
                       C1(x2, rc, out) * C2(rc, x0, x1)

    The partition_rank = ra + rb + rc (sum of individual ranks).

    Usage:
        # With total partition rank (auto-distributed):
        model = PartitionRank3(phys_dim=10, partition_rank=6, output_dim=3)
        # Results in ranks (2, 2, 2)

        # With explicit ranks:
        model = PartitionRank3(phys_dim=10, rank_dims=(3, 2, 1), output_dim=3)
    """

    def __init__(
        self,
        phys_dim: int,
        output_dim: int,
        partition_rank: Optional[int] = None,
        rank_dims: Optional[Tuple[int, int, int]] = None,
        init_strength: float = 0.001,
        use_tn_normalization: bool = True,
        tn_target_std: float = 0.1,
        sample_inputs: Optional[qt.TensorNetwork] = None,
    ):
        """
        Args:
            phys_dim: Physical dimension for each input site (x0, x1, x2)
            output_dim: Output dimension (e.g., number of classes)
            partition_rank: Total partition rank (ra + rb + rc). Auto-distributed across terms.
            rank_dims: Explicit rank dimensions as (ra, rb, rc). Overrides partition_rank.
            init_strength: Base initialization strength (used if use_tn_normalization=False)
            use_tn_normalization: Apply TN normalization after initialization
            tn_target_std: Target standard deviation for TN normalization
            sample_inputs: Sample TN inputs for normalization (optional)
        """
        if rank_dims is not None:
            self.rank_dims = rank_dims
        elif partition_rank is not None:
            self.rank_dims = distribute_partition_rank(partition_rank, n_terms=3)
        else:
            raise ValueError("Must specify either partition_rank or rank_dims")

        self.phys_dim = phys_dim
        self.output_dim = output_dim
        self.partition_rank = sum(self.rank_dims)

        base_init = 0.1 if use_tn_normalization else init_strength

        self.tns: List[qt.TensorNetwork] = []
        self.input_dims_list: List[List[str]] = []
        self.input_labels_list: List[List[str]] = []
        self.output_dims = ["out"]

        input_dims = ["x0", "x1", "x2"]
        input_labels = ["x0", "x1", "x2"]

        term_configs = [
            ("x0", ["x1", "x2"], "ra", "A", self.rank_dims[0]),
            ("x1", ["x0", "x2"], "rb", "B", self.rank_dims[1]),
            ("x2", ["x0", "x1"], "rc", "C", self.rank_dims[2]),
        ]

        for pivot_idx, complement_idxs, rank_idx, term_prefix, rank_dim in term_configs:
            tn = self._create_partition_tn(
                pivot_idx=pivot_idx,
                complement_idxs=complement_idxs,
                rank_idx=rank_idx,
                rank_dim=rank_dim,
                term_prefix=term_prefix,
                base_init=base_init,
            )
            self.tns.append(tn)
            self.input_dims_list.append(input_dims.copy())
            self.input_labels_list.append(input_labels.copy())

        if use_tn_normalization:
            for tn in self.tns:
                self._normalize_tn(tn, sample_inputs, tn_target_std)

    def _create_partition_tn(
        self,
        pivot_idx: str,
        complement_idxs: List[str],
        rank_idx: str,
        rank_dim: int,
        term_prefix: str,
        base_init: float,
    ) -> qt.TensorNetwork:
        """
        Create a single partition term TN.

        Returns TensorNetwork with:
            - {term_prefix}1: (pivot_idx, rank_idx, out) shape (phys_dim, rank_dim, output_dim)
            - {term_prefix}2: (rank_idx, complement1, complement2) shape (rank_dim, phys_dim, phys_dim)
        """
        tensors = []

        t1_data = torch.randn(self.phys_dim, rank_dim, self.output_dim) * base_init
        t1_inds = (pivot_idx, rank_idx, "out")
        t1 = qt.Tensor(data=t1_data, inds=t1_inds, tags={f"{term_prefix}1"})
        tensors.append(t1)

        t2_data = torch.randn(rank_dim, self.phys_dim, self.phys_dim) * base_init
        t2_inds = (rank_idx, complement_idxs[0], complement_idxs[1])
        t2 = qt.Tensor(data=t2_data, inds=t2_inds, tags={f"{term_prefix}2"})
        tensors.append(t2)

        return qt.TensorNetwork(tensors)

    def count_parameters(self) -> int:
        """Count total trainable parameters across all TNs."""
        return sum(t.size for tn in self.tns for t in tn.tensors)

    def parameter_breakdown(self) -> dict:
        """Get parameter count per term."""
        breakdown = {}
        for i, (tn, rank) in enumerate(zip(self.tns, self.rank_dims)):
            term_name = ["A", "B", "C"][i]
            breakdown[term_name] = {"rank": rank, "params": sum(t.size for t in tn.tensors)}
        breakdown["total"] = self.count_parameters()
        breakdown["partition_rank"] = self.partition_rank
        return breakdown

    def _normalize_tn(
        self,
        tn: qt.TensorNetwork,
        sample_inputs: Optional[qt.TensorNetwork],
        tn_target_std: float,
    ):
        """Apply normalization to a tensor network."""
        from model.initialization import normalize_tn_output, normalize_tn_frobenius

        if sample_inputs is not None:
            normalize_tn_output(
                tn,
                sample_inputs,
                output_dims=["out"],
                batch_dim="s",
                target_std=tn_target_std,
            )
        else:
            import numpy as np

            avg_rank = sum(self.rank_dims) / len(self.rank_dims)
            target_norm = np.sqrt(3 * avg_rank * self.phys_dim)
            normalize_tn_frobenius(tn, target_norm=target_norm)
