# type: ignore
"""
NTN_Ensemble: Unified ensemble training for Type I models.

Handles coordination of multiple NTN instances with proper ensemble derivatives:
- f_ensemble = f_1 + f_2 + ... + f_n
- Each model's derivatives are computed w.r.t. ensemble loss
"""

import torch
import quimb.tensor as qt
from typing import List, Dict, Optional, Callable
from model.base.NTN import NTN, NOT_TRAINABLE_TAG
from model.builder import Inputs
from model.exceptions import SingularMatrixError


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
            if hasattr(y_pred_others, "inds") and hasattr(y_pred_self, "inds"):
                if set(y_pred_others.inds) != set(y_pred_self.inds):
                    reindex_map = dict(zip(y_pred_others.inds, y_pred_self.inds))
                    y_pred_others = y_pred_others.reindex(reindex_map)
            result = y_pred_self + y_pred_others
            result.modify(inds=y_pred_self.inds)
            return result
        return y_pred_self

    def reset_batch_idx(self):
        self._batch_idx = 0

    def _batch_node_derivatives(self, inputs, y_true, node_tag):
        result = super()._batch_node_derivatives(inputs, y_true, node_tag)
        self._batch_idx += 1
        return result

    def forward_from_environment(self, env, node_tag=None, node_tensor=None, sum_over_batch=False):
        y_pred_self = super().forward_from_environment(env, node_tag, node_tensor, sum_over_batch)
        y_pred_others = self.get_others_cached_output_fn(self.ntn_index, self._batch_idx)
        if y_pred_others is not None:
            if hasattr(y_pred_others, "inds") and hasattr(y_pred_self, "inds"):
                if set(y_pred_others.inds) != set(y_pred_self.inds):
                    reindex_map = dict(zip(y_pred_others.inds, y_pred_self.inds))
                    y_pred_others = y_pred_others.reindex(reindex_map)
            result = y_pred_self + y_pred_others
            result.modify(inds=y_pred_self.inds)
            return result
        return y_pred_self

    def reset_batch_idx(self):
        self._batch_idx = 0


class NTN_Ensemble:
    """
    Ensemble coordinator for multiple tensor networks with proper ensemble derivatives.

    This class provides the same API as NTN but handles multiple TNs internally.
    Each TN can have different input dimensions (for Type I ensembles with varying L).

    Usage:
        # Create model builder (returns list of TNs with metadata)
        model = MPO2TypeI(max_sites=4, bond_dim=5, phys_dim=10, output_dim=3)

        # Create ensemble (same API as NTN!)
        ntn = NTN_Ensemble(
            tns=model.tns,
            input_dims_list=model.input_dims_list,
            input_labels_list=model.input_labels_list,
            output_dims=model.output_dims,
            loss=loss,
            X_train=X_train,
            y_train=y_train,
        )

        # Train (same API as NTN!)
        ntn.fit(n_epochs=10)
    """

    def __init__(
        self,
        tns: List[qt.TensorNetwork],
        input_dims_list: List[List[str]],
        input_labels_list: List[List[str]],
        output_dims: List[str],
        loss,
        X_train: torch.Tensor,
        y_train: torch.Tensor,
        X_val: Optional[torch.Tensor] = None,
        y_val: Optional[torch.Tensor] = None,
        X_test: Optional[torch.Tensor] = None,
        y_test: Optional[torch.Tensor] = None,
        batch_size: int = 32,
        method: str = "cholesky",
        not_trainable_tags: List[str] = None,
    ):
        """
        Args:
            tns: List of tensor networks (one per ensemble member)
            input_dims_list: List of input_dims for each TN
            input_labels_list: List of input_labels for each TN
            output_dims: Output dimensions (same for all TNs)
            loss: Loss function
            X_train, y_train: Training data
            X_val, y_val: Optional validation data
            X_test, y_test: Optional test data
            batch_size: Batch size for data loaders
            method: Solver method ('cholesky' or 'standard')
            not_trainable_tags: List of tags to mark as non-trainable
        """
        self.n_models = len(tns)
        self.output_dims = output_dims
        self.batch_dim = "s"
        self.loss = loss
        self.singular_encountered = False

        self.ntns: List[NTN_TypeI] = []
        self._forward_cache: Dict[int, List] = {}
        self._forward_cache_val: Dict[int, List] = {}
        self._forward_cache_test: Dict[int, List] = {}

        not_trainable_tags = not_trainable_tags or []

        for idx, (tn, input_dims, input_labels) in enumerate(
            zip(tns, input_dims_list, input_labels_list)
        ):
            train_loader = Inputs(
                inputs=[X_train],
                outputs=[y_train],
                outputs_labels=output_dims,
                input_labels=input_labels,
                batch_dim=self.batch_dim,
                batch_size=batch_size,
            )

            ntn = NTN_TypeI(
                get_others_cached_output_fn=self._get_others_cached_output,
                ntn_index=idx,
                tn=tn,
                output_dims=output_dims,
                input_dims=input_dims,
                loss=loss,
                data_stream=train_loader,
                method=method,
                not_trainable_nodes=not_trainable_tags,
            )

            if X_val is not None and y_val is not None:
                val_loader = Inputs(
                    inputs=[X_val],
                    outputs=[y_val],
                    outputs_labels=output_dims,
                    input_labels=input_labels,
                    batch_dim=self.batch_dim,
                    batch_size=batch_size,
                )
                ntn.val_data = val_loader

            if X_test is not None and y_test is not None:
                test_loader = Inputs(
                    inputs=[X_test],
                    outputs=[y_test],
                    outputs_labels=output_dims,
                    input_labels=input_labels,
                    batch_dim=self.batch_dim,
                    batch_size=batch_size,
                )
                ntn.test_data = test_loader

            self.ntns.append(ntn)

        self.train_data = self.ntns[0].train_data
        self.val_data = self.ntns[0].val_data if X_val is not None else None
        self.test_data = self.ntns[0].test_data if X_test is not None else None

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

    def evaluate(self, metrics, data_stream=None, split: str = None):
        """
        Evaluate ensemble performance.

        Args:
            metrics: Dict of metric functions
            data_stream: Ignored (for API compatibility with NTN)
            split: 'train', 'val', or 'test' (default: 'train')
        """
        if split is None:
            if data_stream is None:
                split = "train"
            elif data_stream == self.val_data:
                split = "val"
            elif data_stream == self.test_data:
                split = "test"
            else:
                split = "train"

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
        data_stream_obj = getattr(ntn, data_attr, None)
        if data_stream_obj is None:
            return {}

        aggregates = {}
        for i, (_, y_true) in enumerate(data_stream_obj.data_mu_y):
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
        jitter=1e-6,
        verbose: bool = True,
        eval_metrics=None,
        val_data=None,
        test_data=None,
        callback_epoch=None,
        callback_init=None,
        patience=None,
        min_delta=0.0,
        adaptive_jitter=False,
        train_selection=False,
    ):
        """
        Train all models with proper ensemble derivatives.

        Same API as NTN.fit() for drop-in compatibility.
        """
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
            print(f"Starting Ensemble Fit: {n_epochs} epochs")
            print(
                f"Models: {[(i + 1, len(ntn._get_trainable_nodes())) for i, ntn in enumerate(self.ntns)]}"
            )
            print(f"Total nodes: {len(all_nodes)}")
            if has_val:
                print("Validation: Using separate validation set")
            if patience is not None:
                print(f"Early stopping: patience={patience}, min_delta={min_delta}")

        scores_train = self.evaluate(eval_metrics, split="train")
        if has_val:
            scores_val = self.evaluate(eval_metrics, split="val")
        else:
            scores_val = scores_train

        if verbose:
            print(f"Init    | Train: ", end="")
            print_metrics(scores_train)
            if has_val:
                print(f"        | Val:   ", end="")
                print_metrics(scores_val)
            print()

        if callback_init is not None:
            info = {
                "n_epochs": n_epochs,
                "jitter_schedule": jitter_schedule,
                "regularize": regularize,
            }
            callback_init(scores_train, scores_val, info)

        best_val_quality = compute_quality(scores_val)
        best_train_quality = compute_quality(scores_train)
        best_scores_train = scores_train.copy()
        best_scores_val = scores_val.copy()
        best_epoch = -1
        patience_counter = 0

        best_tn_states = [ntn.tn.copy() for ntn in self.ntns]

        for epoch in range(n_epochs):
            try:
                self._cache_all_forwards()

                current_ntn_idx = None
                for ntn_idx, node_tag in all_nodes:
                    ntn = self.ntns[ntn_idx]

                    if current_ntn_idx is not None and ntn_idx != current_ntn_idx:
                        self._invalidate_cache(current_ntn_idx)

                    ntn.reset_batch_idx()
                    ntn.update_tn_node(
                        node_tag, regularize, jitter_schedule[epoch], adaptive_jitter
                    )
                    current_ntn_idx = ntn_idx

                if current_ntn_idx is not None:
                    self._invalidate_cache(current_ntn_idx)
            except torch.linalg.LinAlgError as e:
                self.singular_encountered = True
                for ntn, best_state in zip(self.ntns, best_tn_states):
                    ntn.tn = best_state
                if verbose:
                    print(f"\n✗ Singular matrix at epoch {epoch + 1} - stopping training")
                    print(
                        f"  Restoring best model from epoch {best_epoch + 1} (val_quality={best_val_quality:.6f})"
                    )
                raise SingularMatrixError(
                    message="Singular matrix encountered during NTN_Ensemble optimization",
                    epoch=epoch + 1,
                )

            scores_train = self.evaluate(eval_metrics, split="train")

            if has_val:
                scores_val = self.evaluate(eval_metrics, split="val")
            else:
                scores_val = scores_train

            current_val_quality = compute_quality(scores_val)
            current_train_quality = compute_quality(scores_train)

            import math

            if train_selection:
                val_improved = (
                    current_val_quality is not None
                    and math.isfinite(current_val_quality)
                    and current_val_quality > best_val_quality + min_delta
                )
                train_improved = (
                    current_train_quality is not None
                    and math.isfinite(current_train_quality)
                    and current_train_quality > best_train_quality + min_delta
                )
                val_same = (
                    current_val_quality is not None
                    and math.isfinite(current_val_quality)
                    and abs(current_val_quality - best_val_quality) < min_delta
                )

                if val_improved or (val_same and train_improved):
                    best_val_quality = current_val_quality
                    best_train_quality = current_train_quality
                    best_tn_states = [ntn.tn.copy() for ntn in self.ntns]
                    best_scores_train = scores_train.copy()
                    best_scores_val = scores_val.copy()
                    best_epoch = epoch
                    is_best = True
                    patience_counter = 0
                else:
                    is_best = False
                    patience_counter += 1
            else:
                if (
                    current_val_quality is not None
                    and math.isfinite(current_val_quality)
                    and current_val_quality > best_val_quality + min_delta
                ):
                    best_val_quality = current_val_quality
                    best_train_quality = current_train_quality
                    best_tn_states = [ntn.tn.copy() for ntn in self.ntns]
                    best_scores_train = scores_train.copy()
                    best_scores_val = scores_val.copy()
                    best_epoch = epoch
                    is_best = True
                    patience_counter = 0
                else:
                    is_best = False
                    patience_counter += 1

            if verbose:
                marker = " *" if is_best else ""
                print(f"Epoch {epoch + 1} | Train: ", end="")
                print_metrics(scores_train)
                if has_val:
                    print(f"        | Val:   ", end="")
                    print_metrics(scores_val)
                    print(f"{marker}", end="")
                print()

            if callback_epoch is not None:
                info = {
                    "epoch": epoch,
                    "jitter": jitter_schedule[epoch],
                    "regularize": regularize,
                    "reg_loss": scores_train.get("loss", 0),
                    "weight_norm_sq": None,
                    "best_quality": best_val_quality,
                    "is_best": is_best,
                    "patience_counter": patience_counter,
                }
                callback_epoch(epoch, scores_train, scores_val, info)

            if patience and patience_counter >= patience:
                if verbose:
                    print(f"\nEarly stopping at epoch {epoch + 1}")
                    print(f"  Restoring best model from epoch {best_epoch + 1}")
                break

        for ntn, best_state in zip(self.ntns, best_tn_states):
            ntn.tn = best_state

        if verbose and best_epoch >= 0:
            print(
                f"\nRestoring best model from epoch {best_epoch + 1} (val_quality={best_val_quality:.6f})"
            )

        return best_scores_train, best_scores_val
