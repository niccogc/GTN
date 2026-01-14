# type: ignore
"""
MPO2 Type I: Wrapper coordinating multiple NTN instances.

Key insight: f_ensemble = f_1 + f_2 + ... + f_n
Need: dL/dθ_i where L = loss(f_ensemble, y)
Since df_ensemble/dθ_i = df_i/dθ_i, each NTN uses ensemble's dL/df in derivative computation.
"""

import torch
import quimb.tensor as qt
from typing import List, Optional, Callable
from model.NTN import NTN
from model.builder import Inputs


class NTN_TypeI(NTN):
    """NTN that computes ensemble y_pred = f_self + sum(f_others) for proper derivative computation."""

    def __init__(self, get_others_cached_output_fn: Callable, ntn_index: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.get_others_cached_output_fn = get_others_cached_output_fn
        self.ntn_index = ntn_index
        self._batch_idx = 0

    def _batch_node_derivatives(self, inputs, y_true, node_tag):
        result = super()._batch_node_derivatives(inputs, y_true, node_tag)
        self._batch_idx += 1
        return result

    def forward_from_environment(self, env, node_tag=None, node_tensor=None, sum_over_batch=False):
        y_pred_self = super().forward_from_environment(env, node_tag, node_tensor, sum_over_batch)
        y_pred_others = self.get_others_cached_output_fn(self.ntn_index, self._batch_idx)
        if y_pred_others is not None:
            return y_pred_self + y_pred_others
        return y_pred_self

    def reset_batch_idx(self):
        self._batch_idx = 0


def create_simple_mps(
    L: int,
    bond_dim: int,
    phys_dim: int,
    output_dim: int,
    output_site: Optional[int] = None,
    init_strength: float = 0.001,
):
    """Create simple MPS. For L=1: just (phys_dim, output_dim) with no bonds."""
    if output_site is None:
        output_site = L - 1

    tensors = []

    if L == 1:
        shape = (phys_dim, output_dim)
        inds = ("x0", "out")
        data = torch.randn(*shape) * init_strength
        tensor = qt.Tensor(data=data, inds=inds, tags={"Node0"})
        tensors.append(tensor)
    else:
        for i in range(L):
            if i == 0:
                shape = (phys_dim, bond_dim)
                inds = (f"x{i}", f"b{i}")
            elif i == L - 1:
                shape = (bond_dim, phys_dim)
                inds = (f"b{i - 1}", f"x{i}")
            else:
                shape = (bond_dim, phys_dim, bond_dim)
                inds = (f"b{i - 1}", f"x{i}", f"b{i}")

            if i == output_site:
                shape = shape + (output_dim,)
                inds = inds + ("out",)

            data = torch.randn(*shape) * init_strength
            tensor = qt.Tensor(data=data, inds=inds, tags={f"Node{i}"})
            tensors.append(tensor)

    tn = qt.TensorNetwork(tensors)
    input_labels = [f"x{i}" for i in range(L)]
    input_dims = input_labels
    output_dims = ["out"]

    return tn, input_labels, input_dims, output_dims


class MPO2TypeI:
    """Coordinator for multiple NTN_TypeI instances with proper ensemble derivatives."""

    def __init__(
        self,
        max_sites: int,
        bond_dim: int,
        phys_dim: int,
        output_dim: int,
        loss,
        X_train: torch.Tensor,
        y_train: torch.Tensor,
        X_val: torch.Tensor = None,
        y_val: torch.Tensor = None,
        X_test: torch.Tensor = None,
        y_test: torch.Tensor = None,
        batch_size: int = 32,
        output_site: Optional[int] = None,
        init_strength: float = 0.001,
        method: str = "cholesky",
        not_trainable_nodes: List[str] = [],
    ):
        self.max_sites = max_sites
        self.ntns = []
        self._forward_cache = {}
        self._forward_cache_val = {}
        self._forward_cache_test = {}

        for L in range(1, max_sites + 1):
            tn, input_labels, input_dims, output_dims = create_simple_mps(
                L=L,
                bond_dim=bond_dim,
                phys_dim=phys_dim,
                output_dim=output_dim,
                output_site=output_site,
                init_strength=init_strength,
            )

            train_loader = Inputs(
                inputs=[X_train],
                outputs=[y_train],
                outputs_labels=output_dims,
                input_labels=input_labels,
                batch_dim="s",
                batch_size=batch_size,
            )

            ntn_index = L - 1
            ntn = NTN_TypeI(
                get_others_cached_output_fn=self._get_others_cached_output,
                ntn_index=ntn_index,
                tn=tn,
                output_dims=output_dims,
                input_dims=input_dims,
                loss=loss,
                data_stream=train_loader,
                method=method,
                not_trainable_nodes=not_trainable_nodes,
            )

            if X_val is not None and y_val is not None:
                val_loader = Inputs(
                    inputs=[X_val],
                    outputs=[y_val],
                    outputs_labels=output_dims,
                    input_labels=input_labels,
                    batch_dim="s",
                    batch_size=batch_size,
                )
                ntn.val_data = val_loader

            if X_test is not None and y_test is not None:
                test_loader = Inputs(
                    inputs=[X_test],
                    outputs=[y_test],
                    outputs_labels=output_dims,
                    input_labels=input_labels,
                    batch_dim="s",
                    batch_size=batch_size,
                )
                ntn.test_data = test_loader

            self.ntns.append(ntn)

        self.loss = loss
        self.output_dims = output_dims
        self.batch_dim = "s"

    def _cache_forwards_for_data(self, data_attr: str, cache_dict: dict):
        """Cache forward outputs for all models using specified data attribute."""
        cache_dict.clear()
        for ntn_idx, ntn in enumerate(self.ntns):
            data_stream = getattr(ntn, data_attr, None)
            if data_stream is None:
                continue
            batch_outputs = []
            for inputs in data_stream.data_mu:
                y_pred = ntn._batch_forward(inputs, ntn.tn, [self.batch_dim] + self.output_dims)
                batch_outputs.append(y_pred)
            cache_dict[ntn_idx] = batch_outputs

    def _cache_all_forwards(self):
        """Cache forward outputs for training data."""
        self._cache_forwards_for_data("data", self._forward_cache)

    def _cache_val_forwards(self):
        """Cache forward outputs for validation data."""
        self._cache_forwards_for_data("val_data", self._forward_cache_val)

    def _cache_test_forwards(self):
        """Cache forward outputs for test data."""
        self._cache_forwards_for_data("test_data", self._forward_cache_test)

    def _invalidate_cache(self, ntn_idx: int):
        """Recompute cache for a specific model after its nodes are updated."""
        ntn = self.ntns[ntn_idx]
        batch_outputs = []
        for inputs in ntn.data.data_mu:
            y_pred = ntn._batch_forward(inputs, ntn.tn, [self.batch_dim] + self.output_dims)
            batch_outputs.append(y_pred)
        self._forward_cache[ntn_idx] = batch_outputs

    def _get_others_cached_output(self, exclude_idx: int, batch_idx: int):
        """Get sum of cached outputs from all models except exclude_idx for given batch."""
        total = None
        for ntn_idx, batch_outputs in self._forward_cache.items():
            if ntn_idx == exclude_idx:
                continue
            if batch_idx < len(batch_outputs):
                if total is None:
                    total = batch_outputs[batch_idx]
                else:
                    total = total + batch_outputs[batch_idx]
        return total

    def _get_all_trainable_nodes(self) -> List[tuple]:
        """Get all trainable nodes as (ntn_idx, node_tag) tuples."""
        all_nodes = []
        for ntn_idx, ntn in enumerate(self.ntns):
            trainable = ntn._get_trainable_nodes()
            for tag in trainable:
                all_nodes.append((ntn_idx, tag))
        return all_nodes

    def forward(self) -> qt.Tensor:
        """Ensemble forward: sum outputs from all NTN instances using their own data streams."""
        total = None
        for ntn in self.ntns:
            y_pred = ntn.forward(
                ntn.tn, ntn.data.data_mu, sum_over_batch=False, sum_over_output=False
            )
            if total is None:
                total = y_pred
            else:
                total = total + y_pred
        return total

    def _to_torch(self, x):
        if hasattr(x, "data"):
            return x.data
        return x

    def _batch_evaluate(self, batch_idx, y_true, metrics, cache_dict):
        """Evaluate one batch using ensemble forward from cached outputs."""
        y_pred = None
        for ntn_idx, batch_outputs in cache_dict.items():
            if batch_idx < len(batch_outputs):
                if y_pred is None:
                    y_pred = batch_outputs[batch_idx]
                else:
                    y_pred = y_pred + batch_outputs[batch_idx]

        y_pred_th = self._to_torch(y_pred)
        y_true_th = self._to_torch(y_true)

        results = {}
        for name, func in metrics.items():
            results[name] = func(y_pred_th, y_true_th)
        return results

    def _aggregate_scores(self, aggregates):
        """Convert aggregated metrics to final scores."""
        final_scores = {}
        for name, val in aggregates.items():
            if isinstance(val, (tuple, list)) and len(val) == 2:
                numerator, denominator = val
                if torch.is_tensor(denominator):
                    denominator = denominator.item()
                if denominator != 0:
                    final_scores[name] = (
                        (numerator / denominator).item()
                        if torch.is_tensor(numerator)
                        else (numerator / denominator)
                    )
                else:
                    final_scores[name] = 0.0
            else:
                final_scores[name] = val
        return final_scores

    def evaluate(self, metrics, split: str = "train"):
        """Evaluate ensemble performance on train/val/test split."""
        if split == "train":
            self._cache_all_forwards()
            cache_dict = self._forward_cache
            data_attr = "data"
        elif split == "val":
            self._cache_val_forwards()
            cache_dict = self._forward_cache_val
            data_attr = "val_data"
        elif split == "test":
            self._cache_test_forwards()
            cache_dict = self._forward_cache_test
            data_attr = "test_data"
        else:
            raise ValueError(f"Unknown split: {split}")

        ntn = self.ntns[0]
        data_stream = getattr(ntn, data_attr, None)
        if data_stream is None:
            return {}

        aggregates = {}
        for i, (_, y_true) in enumerate(data_stream.data_mu_y):
            batch_results = self._batch_evaluate(i, y_true, metrics, cache_dict)

            if i == 0:
                aggregates = batch_results
            else:
                for name, res in batch_results.items():
                    if isinstance(res, tuple):
                        aggregates[name] = tuple(a + b for a, b in zip(aggregates[name], res))
                    else:
                        aggregates[name] += res

        return self._aggregate_scores(aggregates)

    def fit(
        self,
        n_epochs: int = 1,
        regularize: bool = True,
        jitter: float = 1e-6,
        verbose: bool = True,
        eval_metrics=None,
        patience=None,
        min_delta=0.0,
        adaptive_jitter=False,
    ):
        """Train all models by simply calling each NTN's update method."""
        from model.utils import REGRESSION_METRICS, compute_quality, print_metrics

        if eval_metrics is None:
            eval_metrics = REGRESSION_METRICS

        if not isinstance(jitter, list):
            jitter_schedule = [jitter] * n_epochs
        else:
            jitter_schedule = jitter

        all_nodes = self._get_all_trainable_nodes()
        has_val = self.ntns[0].val_data is not None

        if verbose:
            print(f"Starting Fit: {n_epochs} epochs")
            print(
                f"Models: {[(i + 1, len(ntn._get_trainable_nodes())) for i, ntn in enumerate(self.ntns)]}"
            )
            print(f"Total nodes: {len(all_nodes)}")

        best_val_quality = float("-inf")
        patience_counter = 0

        for epoch in range(n_epochs):
            self._cache_all_forwards()

            current_ntn_idx = None
            for ntn_idx, node_tag in all_nodes:
                ntn = self.ntns[ntn_idx]

                if current_ntn_idx is not None and ntn_idx != current_ntn_idx:
                    self._invalidate_cache(current_ntn_idx)

                ntn.reset_batch_idx()
                ntn.update_tn_node(node_tag, regularize, jitter_schedule[epoch], adaptive_jitter)
                current_ntn_idx = ntn_idx

            if current_ntn_idx is not None:
                self._invalidate_cache(current_ntn_idx)

            scores_train = self.evaluate(eval_metrics, split="train")

            if has_val:
                scores_val = self.evaluate(eval_metrics, split="val")
            else:
                scores_val = scores_train

            current_val_quality = compute_quality(scores_val)

            if current_val_quality > best_val_quality + min_delta:
                best_val_quality = current_val_quality
                patience_counter = 0
                is_best = True
            else:
                patience_counter += 1
                is_best = False

            if verbose:
                marker = " *" if is_best else ""
                print(f"Epoch {epoch + 1} | Train: ", end="")
                print_metrics(scores_train)
                if has_val:
                    print(f"        | Val:   ", end="")
                    print_metrics(scores_val)
                    print(f"{marker}", end="")
                print()

            if patience and patience_counter >= patience:
                if verbose:
                    print(f"Early stopping at epoch {epoch + 1}")
                break

        return scores_train, scores_val if has_val else scores_train


def create_lmpo2(
    L: int,
    bond_dim: int,
    phys_dim: int,
    reduced_dim: int,
    output_dim: int,
    output_site: Optional[int] = None,
    init_strength: float = 0.001,
):
    """Create LMPO2: MPO (trainable) for reduction + MPS (trainable) for output."""
    if output_site is None:
        output_site = L - 1

    tensors = []

    if L == 1:
        mpo_data = torch.randn(phys_dim, reduced_dim) * init_strength
        mpo_tensor = qt.Tensor(data=mpo_data, inds=("0_in", "0_reduced"), tags={"0_MPO"})
        tensors.append(mpo_tensor)

        mps_data = torch.randn(reduced_dim, output_dim) * init_strength
        mps_tensor = qt.Tensor(data=mps_data, inds=("0_reduced", "out"), tags={"0_MPS"})
        tensors.append(mps_tensor)
    else:
        for i in range(L):
            if i == 0:
                data = torch.randn(phys_dim, reduced_dim, bond_dim) * init_strength
                inds = (f"{i}_in", f"{i}_reduced", f"b_mpo_{i}")
            elif i == L - 1:
                data = torch.randn(bond_dim, phys_dim, reduced_dim) * init_strength
                inds = (f"b_mpo_{i - 1}", f"{i}_in", f"{i}_reduced")
            else:
                data = torch.randn(bond_dim, phys_dim, reduced_dim, bond_dim) * init_strength
                inds = (f"b_mpo_{i - 1}", f"{i}_in", f"{i}_reduced", f"b_mpo_{i}")
            tensors.append(qt.Tensor(data=data, inds=inds, tags={f"{i}_MPO"}))

        for i in range(L):
            if i == 0:
                shape = (reduced_dim, bond_dim)
                inds = (f"{i}_reduced", f"b_mps_{i}")
            elif i == L - 1:
                shape = (bond_dim, reduced_dim)
                inds = (f"b_mps_{i - 1}", f"{i}_reduced")
            else:
                shape = (bond_dim, reduced_dim, bond_dim)
                inds = (f"b_mps_{i - 1}", f"{i}_reduced", f"b_mps_{i}")

            if i == output_site:
                shape = shape + (output_dim,)
                inds = inds + ("out",)

            data = torch.randn(*shape) * init_strength
            tensors.append(qt.Tensor(data=data, inds=inds, tags={f"{i}_MPS"}))

    tn = qt.TensorNetwork(tensors)
    input_labels = [f"{i}_in" for i in range(L)]
    input_dims = input_labels
    output_dims = ["out"]

    return tn, input_labels, input_dims, output_dims


def create_mmpo2(
    L: int,
    bond_dim: int,
    phys_dim: int,
    output_dim: int,
    output_site: Optional[int] = None,
    init_strength: float = 0.001,
):
    """Create MMPO2: MPO mask (NOT trainable) + MPS (trainable) for output."""
    if output_site is None:
        output_site = L - 1

    tensors = []
    mask_bond_dim = phys_dim

    H = torch.zeros(phys_dim, phys_dim)
    for i in range(phys_dim):
        for j in range(phys_dim):
            H[i, j] = 1.0 if j >= i else 0.0

    if L == 1:
        mask_data = torch.eye(phys_dim)
        mask_tensor = qt.Tensor(data=mask_data, inds=("0_in", "0_masked"), tags={"0_Mask", "NT"})
        tensors.append(mask_tensor)

        mps_data = torch.randn(phys_dim, output_dim) * init_strength
        mps_tensor = qt.Tensor(data=mps_data, inds=("0_masked", "out"), tags={"0_MPS"})
        tensors.append(mps_tensor)
    else:
        for i in range(L):
            if i == 0:
                Delta = torch.zeros(phys_dim, phys_dim, mask_bond_dim)
                for k in range(phys_dim):
                    Delta[k, k, k] = 1.0
                data = Delta
                inds = (f"{i}_in", f"{i}_masked", f"b_mask_{i}")
            elif i == L - 1:
                Delta = torch.zeros(mask_bond_dim, phys_dim, phys_dim)
                for k in range(mask_bond_dim):
                    Delta[k, k, k] = 1.0
                data = torch.einsum("bk,kio->bio", H, Delta)
                inds = (f"b_mask_{i - 1}", f"{i}_in", f"{i}_masked")
            else:
                Delta = torch.zeros(mask_bond_dim, phys_dim, phys_dim, mask_bond_dim)
                for k in range(mask_bond_dim):
                    Delta[k, k, k, k] = 1.0
                data = torch.einsum("bk,kior->bior", H, Delta)
                inds = (f"b_mask_{i - 1}", f"{i}_in", f"{i}_masked", f"b_mask_{i}")
            tensors.append(qt.Tensor(data=data, inds=inds, tags={f"{i}_Mask", "NT"}))

        for i in range(L):
            if i == 0:
                shape = (phys_dim, bond_dim)
                inds = (f"{i}_masked", f"b_mps_{i}")
            elif i == L - 1:
                shape = (bond_dim, phys_dim)
                inds = (f"b_mps_{i - 1}", f"{i}_masked")
            else:
                shape = (bond_dim, phys_dim, bond_dim)
                inds = (f"b_mps_{i - 1}", f"{i}_masked", f"b_mps_{i}")

            if i == output_site:
                shape = shape + (output_dim,)
                inds = inds + ("out",)

            data = torch.randn(*shape) * init_strength
            tensors.append(qt.Tensor(data=data, inds=inds, tags={f"{i}_MPS"}))

    tn = qt.TensorNetwork(tensors)
    input_labels = [f"{i}_in" for i in range(L)]
    input_dims = input_labels
    output_dims = ["out"]

    return tn, input_labels, input_dims, output_dims


class LMPO2TypeI:
    """Type I ensemble for LMPO2: MPO reduction + MPS output, varying L."""

    def __init__(
        self,
        max_sites: int,
        bond_dim: int,
        phys_dim: int,
        reduced_dim: int,
        output_dim: int,
        loss,
        X_train: torch.Tensor,
        y_train: torch.Tensor,
        X_val: torch.Tensor = None,
        y_val: torch.Tensor = None,
        X_test: torch.Tensor = None,
        y_test: torch.Tensor = None,
        batch_size: int = 32,
        output_site: Optional[int] = None,
        init_strength: float = 0.001,
        method: str = "cholesky",
        not_trainable_nodes: List[str] = [],
    ):
        self.max_sites = max_sites
        self.ntns = []
        self._forward_cache = {}
        self._forward_cache_val = {}
        self._forward_cache_test = {}

        for L in range(1, max_sites + 1):
            tn, input_labels, input_dims, output_dims = create_lmpo2(
                L=L,
                bond_dim=bond_dim,
                phys_dim=phys_dim,
                reduced_dim=reduced_dim,
                output_dim=output_dim,
                output_site=output_site,
                init_strength=init_strength,
            )

            train_loader = Inputs(
                inputs=[X_train],
                outputs=[y_train],
                outputs_labels=output_dims,
                input_labels=input_labels,
                batch_dim="s",
                batch_size=batch_size,
            )

            ntn_index = L - 1
            ntn = NTN_TypeI(
                get_others_cached_output_fn=self._get_others_cached_output,
                ntn_index=ntn_index,
                tn=tn,
                output_dims=output_dims,
                input_dims=input_dims,
                loss=loss,
                data_stream=train_loader,
                method=method,
                not_trainable_nodes=not_trainable_nodes,
            )

            if X_val is not None and y_val is not None:
                val_loader = Inputs(
                    inputs=[X_val],
                    outputs=[y_val],
                    outputs_labels=output_dims,
                    input_labels=input_labels,
                    batch_dim="s",
                    batch_size=batch_size,
                )
                ntn.val_data = val_loader

            if X_test is not None and y_test is not None:
                test_loader = Inputs(
                    inputs=[X_test],
                    outputs=[y_test],
                    outputs_labels=output_dims,
                    input_labels=input_labels,
                    batch_dim="s",
                    batch_size=batch_size,
                )
                ntn.test_data = test_loader

            self.ntns.append(ntn)

        self.loss = loss
        self.output_dims = output_dims
        self.batch_dim = "s"

    def _cache_forwards_for_data(self, data_attr: str, cache_dict: dict):
        cache_dict.clear()
        for ntn_idx, ntn in enumerate(self.ntns):
            data_stream = getattr(ntn, data_attr, None)
            if data_stream is None:
                continue
            batch_outputs = []
            for inputs in data_stream.data_mu:
                y_pred = ntn._batch_forward(inputs, ntn.tn, [self.batch_dim] + self.output_dims)
                batch_outputs.append(y_pred)
            cache_dict[ntn_idx] = batch_outputs

    def _cache_all_forwards(self):
        self._cache_forwards_for_data("data", self._forward_cache)

    def _cache_val_forwards(self):
        self._cache_forwards_for_data("val_data", self._forward_cache_val)

    def _cache_test_forwards(self):
        self._cache_forwards_for_data("test_data", self._forward_cache_test)

    def _invalidate_cache(self, ntn_idx: int):
        ntn = self.ntns[ntn_idx]
        batch_outputs = []
        for inputs in ntn.data.data_mu:
            y_pred = ntn._batch_forward(inputs, ntn.tn, [self.batch_dim] + self.output_dims)
            batch_outputs.append(y_pred)
        self._forward_cache[ntn_idx] = batch_outputs

    def _get_others_cached_output(self, exclude_idx: int, batch_idx: int):
        total = None
        for ntn_idx, batch_outputs in self._forward_cache.items():
            if ntn_idx == exclude_idx:
                continue
            if batch_idx < len(batch_outputs):
                if total is None:
                    total = batch_outputs[batch_idx]
                else:
                    total = total + batch_outputs[batch_idx]
        return total

    def _get_all_trainable_nodes(self) -> List[tuple]:
        all_nodes = []
        for ntn_idx, ntn in enumerate(self.ntns):
            trainable = ntn._get_trainable_nodes()
            for tag in trainable:
                all_nodes.append((ntn_idx, tag))
        return all_nodes

    def _to_torch(self, x):
        if hasattr(x, "data"):
            return x.data
        return x

    def _batch_evaluate(self, batch_idx, y_true, metrics, cache_dict):
        y_pred = None
        for ntn_idx, batch_outputs in cache_dict.items():
            if batch_idx < len(batch_outputs):
                if y_pred is None:
                    y_pred = batch_outputs[batch_idx]
                else:
                    y_pred = y_pred + batch_outputs[batch_idx]

        y_pred_th = self._to_torch(y_pred)
        y_true_th = self._to_torch(y_true)

        results = {}
        for name, func in metrics.items():
            results[name] = func(y_pred_th, y_true_th)
        return results

    def _aggregate_scores(self, aggregates):
        final_scores = {}
        for name, val in aggregates.items():
            if isinstance(val, (tuple, list)) and len(val) == 2:
                numerator, denominator = val
                if torch.is_tensor(denominator):
                    denominator = denominator.item()
                if denominator != 0:
                    final_scores[name] = (
                        (numerator / denominator).item()
                        if torch.is_tensor(numerator)
                        else (numerator / denominator)
                    )
                else:
                    final_scores[name] = 0.0
            else:
                final_scores[name] = val
        return final_scores

    def evaluate(self, metrics, split: str = "train"):
        if split == "train":
            self._cache_all_forwards()
            cache_dict = self._forward_cache
            data_attr = "data"
        elif split == "val":
            self._cache_val_forwards()
            cache_dict = self._forward_cache_val
            data_attr = "val_data"
        elif split == "test":
            self._cache_test_forwards()
            cache_dict = self._forward_cache_test
            data_attr = "test_data"
        else:
            raise ValueError(f"Unknown split: {split}")

        ntn = self.ntns[0]
        data_stream = getattr(ntn, data_attr, None)
        if data_stream is None:
            return {}

        aggregates = {}
        for i, (_, y_true) in enumerate(data_stream.data_mu_y):
            batch_results = self._batch_evaluate(i, y_true, metrics, cache_dict)

            if i == 0:
                aggregates = batch_results
            else:
                for name, res in batch_results.items():
                    if isinstance(res, tuple):
                        aggregates[name] = tuple(a + b for a, b in zip(aggregates[name], res))
                    else:
                        aggregates[name] += res

        return self._aggregate_scores(aggregates)

    def fit(
        self,
        n_epochs: int = 1,
        regularize: bool = True,
        jitter: float = 1e-6,
        verbose: bool = True,
        eval_metrics=None,
        patience=None,
        min_delta=0.0,
        adaptive_jitter=False,
    ):
        from model.utils import REGRESSION_METRICS, compute_quality, print_metrics

        if eval_metrics is None:
            eval_metrics = REGRESSION_METRICS

        if not isinstance(jitter, list):
            jitter_schedule = [jitter] * n_epochs
        else:
            jitter_schedule = jitter

        all_nodes = self._get_all_trainable_nodes()
        has_val = self.ntns[0].val_data is not None

        if verbose:
            print(f"Starting LMPO2TypeI Fit: {n_epochs} epochs")
            print(
                f"Models: {[(i + 1, len(ntn._get_trainable_nodes())) for i, ntn in enumerate(self.ntns)]}"
            )
            print(f"Total nodes: {len(all_nodes)}")

        best_val_quality = float("-inf")
        patience_counter = 0

        for epoch in range(n_epochs):
            self._cache_all_forwards()

            current_ntn_idx = None
            for ntn_idx, node_tag in all_nodes:
                ntn = self.ntns[ntn_idx]

                if current_ntn_idx is not None and ntn_idx != current_ntn_idx:
                    self._invalidate_cache(current_ntn_idx)

                ntn.reset_batch_idx()
                ntn.update_tn_node(node_tag, regularize, jitter_schedule[epoch], adaptive_jitter)
                current_ntn_idx = ntn_idx

            if current_ntn_idx is not None:
                self._invalidate_cache(current_ntn_idx)

            scores_train = self.evaluate(eval_metrics, split="train")

            if has_val:
                scores_val = self.evaluate(eval_metrics, split="val")
            else:
                scores_val = scores_train

            current_val_quality = compute_quality(scores_val)

            if current_val_quality > best_val_quality + min_delta:
                best_val_quality = current_val_quality
                patience_counter = 0
                is_best = True
            else:
                patience_counter += 1
                is_best = False

            if verbose:
                marker = " *" if is_best else ""
                print(f"Epoch {epoch + 1} | Train: ", end="")
                print_metrics(scores_train)
                if has_val:
                    print(f"        | Val:   ", end="")
                    print_metrics(scores_val)
                    print(f"{marker}", end="")
                print()

            if patience and patience_counter >= patience:
                if verbose:
                    print(f"Early stopping at epoch {epoch + 1}")
                break

        return scores_train, scores_val if has_val else scores_train


class MMPO2TypeI:
    """Type I ensemble for MMPO2: MPO mask (non-trainable) + MPS output, varying L."""

    def __init__(
        self,
        max_sites: int,
        bond_dim: int,
        phys_dim: int,
        output_dim: int,
        loss,
        X_train: torch.Tensor,
        y_train: torch.Tensor,
        X_val: torch.Tensor = None,
        y_val: torch.Tensor = None,
        X_test: torch.Tensor = None,
        y_test: torch.Tensor = None,
        batch_size: int = 32,
        output_site: Optional[int] = None,
        init_strength: float = 0.001,
        method: str = "cholesky",
    ):
        self.max_sites = max_sites
        self.ntns = []
        self._forward_cache = {}
        self._forward_cache_val = {}
        self._forward_cache_test = {}

        for L in range(1, max_sites + 1):
            tn, input_labels, input_dims, output_dims = create_mmpo2(
                L=L,
                bond_dim=bond_dim,
                phys_dim=phys_dim,
                output_dim=output_dim,
                output_site=output_site,
                init_strength=init_strength,
            )

            train_loader = Inputs(
                inputs=[X_train],
                outputs=[y_train],
                outputs_labels=output_dims,
                input_labels=input_labels,
                batch_dim="s",
                batch_size=batch_size,
            )

            ntn_index = L - 1
            ntn = NTN_TypeI(
                get_others_cached_output_fn=self._get_others_cached_output,
                ntn_index=ntn_index,
                tn=tn,
                output_dims=output_dims,
                input_dims=input_dims,
                loss=loss,
                data_stream=train_loader,
                method=method,
                not_trainable_nodes=[f"{i}_Mask" for i in range(L)],
            )

            if X_val is not None and y_val is not None:
                val_loader = Inputs(
                    inputs=[X_val],
                    outputs=[y_val],
                    outputs_labels=output_dims,
                    input_labels=input_labels,
                    batch_dim="s",
                    batch_size=batch_size,
                )
                ntn.val_data = val_loader

            if X_test is not None and y_test is not None:
                test_loader = Inputs(
                    inputs=[X_test],
                    outputs=[y_test],
                    outputs_labels=output_dims,
                    input_labels=input_labels,
                    batch_dim="s",
                    batch_size=batch_size,
                )
                ntn.test_data = test_loader

            self.ntns.append(ntn)

        self.loss = loss
        self.output_dims = output_dims
        self.batch_dim = "s"

    def _cache_forwards_for_data(self, data_attr: str, cache_dict: dict):
        cache_dict.clear()
        for ntn_idx, ntn in enumerate(self.ntns):
            data_stream = getattr(ntn, data_attr, None)
            if data_stream is None:
                continue
            batch_outputs = []
            for inputs in data_stream.data_mu:
                y_pred = ntn._batch_forward(inputs, ntn.tn, [self.batch_dim] + self.output_dims)
                batch_outputs.append(y_pred)
            cache_dict[ntn_idx] = batch_outputs

    def _cache_all_forwards(self):
        self._cache_forwards_for_data("data", self._forward_cache)

    def _cache_val_forwards(self):
        self._cache_forwards_for_data("val_data", self._forward_cache_val)

    def _cache_test_forwards(self):
        self._cache_forwards_for_data("test_data", self._forward_cache_test)

    def _invalidate_cache(self, ntn_idx: int):
        ntn = self.ntns[ntn_idx]
        batch_outputs = []
        for inputs in ntn.data.data_mu:
            y_pred = ntn._batch_forward(inputs, ntn.tn, [self.batch_dim] + self.output_dims)
            batch_outputs.append(y_pred)
        self._forward_cache[ntn_idx] = batch_outputs

    def _get_others_cached_output(self, exclude_idx: int, batch_idx: int):
        total = None
        for ntn_idx, batch_outputs in self._forward_cache.items():
            if ntn_idx == exclude_idx:
                continue
            if batch_idx < len(batch_outputs):
                if total is None:
                    total = batch_outputs[batch_idx]
                else:
                    total = total + batch_outputs[batch_idx]
        return total

    def _get_all_trainable_nodes(self) -> List[tuple]:
        all_nodes = []
        for ntn_idx, ntn in enumerate(self.ntns):
            trainable = ntn._get_trainable_nodes()
            for tag in trainable:
                all_nodes.append((ntn_idx, tag))
        return all_nodes

    def _to_torch(self, x):
        if hasattr(x, "data"):
            return x.data
        return x

    def _batch_evaluate(self, batch_idx, y_true, metrics, cache_dict):
        y_pred = None
        for ntn_idx, batch_outputs in cache_dict.items():
            if batch_idx < len(batch_outputs):
                if y_pred is None:
                    y_pred = batch_outputs[batch_idx]
                else:
                    y_pred = y_pred + batch_outputs[batch_idx]

        y_pred_th = self._to_torch(y_pred)
        y_true_th = self._to_torch(y_true)

        results = {}
        for name, func in metrics.items():
            results[name] = func(y_pred_th, y_true_th)
        return results

    def _aggregate_scores(self, aggregates):
        final_scores = {}
        for name, val in aggregates.items():
            if isinstance(val, (tuple, list)) and len(val) == 2:
                numerator, denominator = val
                if torch.is_tensor(denominator):
                    denominator = denominator.item()
                if denominator != 0:
                    final_scores[name] = (
                        (numerator / denominator).item()
                        if torch.is_tensor(numerator)
                        else (numerator / denominator)
                    )
                else:
                    final_scores[name] = 0.0
            else:
                final_scores[name] = val
        return final_scores

    def evaluate(self, metrics, split: str = "train"):
        if split == "train":
            self._cache_all_forwards()
            cache_dict = self._forward_cache
            data_attr = "data"
        elif split == "val":
            self._cache_val_forwards()
            cache_dict = self._forward_cache_val
            data_attr = "val_data"
        elif split == "test":
            self._cache_test_forwards()
            cache_dict = self._forward_cache_test
            data_attr = "test_data"
        else:
            raise ValueError(f"Unknown split: {split}")

        ntn = self.ntns[0]
        data_stream = getattr(ntn, data_attr, None)
        if data_stream is None:
            return {}

        aggregates = {}
        for i, (_, y_true) in enumerate(data_stream.data_mu_y):
            batch_results = self._batch_evaluate(i, y_true, metrics, cache_dict)

            if i == 0:
                aggregates = batch_results
            else:
                for name, res in batch_results.items():
                    if isinstance(res, tuple):
                        aggregates[name] = tuple(a + b for a, b in zip(aggregates[name], res))
                    else:
                        aggregates[name] += res

        return self._aggregate_scores(aggregates)

    def fit(
        self,
        n_epochs: int = 1,
        regularize: bool = True,
        jitter: float = 1e-6,
        verbose: bool = True,
        eval_metrics=None,
        patience=None,
        min_delta=0.0,
        adaptive_jitter=False,
    ):
        from model.utils import REGRESSION_METRICS, compute_quality, print_metrics

        if eval_metrics is None:
            eval_metrics = REGRESSION_METRICS

        if not isinstance(jitter, list):
            jitter_schedule = [jitter] * n_epochs
        else:
            jitter_schedule = jitter

        all_nodes = self._get_all_trainable_nodes()
        has_val = self.ntns[0].val_data is not None

        if verbose:
            print(f"Starting MMPO2TypeI Fit: {n_epochs} epochs")
            print(
                f"Models: {[(i + 1, len(ntn._get_trainable_nodes())) for i, ntn in enumerate(self.ntns)]}"
            )
            print(f"Total nodes: {len(all_nodes)}")

        best_val_quality = float("-inf")
        patience_counter = 0

        for epoch in range(n_epochs):
            self._cache_all_forwards()

            current_ntn_idx = None
            for ntn_idx, node_tag in all_nodes:
                ntn = self.ntns[ntn_idx]

                if current_ntn_idx is not None and ntn_idx != current_ntn_idx:
                    self._invalidate_cache(current_ntn_idx)

                ntn.reset_batch_idx()
                ntn.update_tn_node(node_tag, regularize, jitter_schedule[epoch], adaptive_jitter)
                current_ntn_idx = ntn_idx

            if current_ntn_idx is not None:
                self._invalidate_cache(current_ntn_idx)

            scores_train = self.evaluate(eval_metrics, split="train")

            if has_val:
                scores_val = self.evaluate(eval_metrics, split="val")
            else:
                scores_val = scores_train

            current_val_quality = compute_quality(scores_val)

            if current_val_quality > best_val_quality + min_delta:
                best_val_quality = current_val_quality
                patience_counter = 0
                is_best = True
            else:
                patience_counter += 1
                is_best = False

            if verbose:
                marker = " *" if is_best else ""
                print(f"Epoch {epoch + 1} | Train: ", end="")
                print_metrics(scores_train)
                if has_val:
                    print(f"        | Val:   ", end="")
                    print_metrics(scores_val)
                    print(f"{marker}", end="")
                print()

            if patience and patience_counter >= patience:
                if verbose:
                    print(f"Early stopping at epoch {epoch + 1}")
                break

        return scores_train, scores_val if has_val else scores_train
