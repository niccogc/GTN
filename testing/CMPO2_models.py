# type: ignore
"""
CMPO2 (Cross MPO2) Models: Two-level tensor network structure.

Structure:
- Pixel level MPS: processes pixels within patches
- Patch level MPS: processes patch-level features
- Input format: (batch, n_patches, pixels_per_patch)

Models:
- CMPO2: Standard model for NTN
- CMPO2_GTN: GTN wrapper with construct_nodes
- CMPO2TypeI: Ensemble of CMPO2 with varying n_sites
"""

import torch
import quimb.tensor as qt
from typing import List, Optional
from model.base.GTN import GTN


def _create_cmpo2(
    n_sites: int,
    pixel_dim: int,
    patch_dim: int,
    rank_pixel: int,
    rank_patch: int,
    output_dim: int,
    output_site: Optional[int] = None,
    init_strength: float = 0.01,
):
    if output_site is None:
        output_site = n_sites - 1

    tensors = []

    if n_sites == 1:
        pixel_node = qt.Tensor(
            torch.randn(pixel_dim) * init_strength, inds=("pixel_0",), tags={"PixelNode_0"}
        )
        tensors.append(pixel_node)

        patch_node = qt.Tensor(
            torch.randn(patch_dim, output_dim) * init_strength,
            inds=("patch_0", "out"),
            tags={"PatchNode_0"},
        )
        tensors.append(patch_node)
    else:
        for i in range(n_sites):
            if i == 0:
                pixel_shape = (pixel_dim, rank_pixel)
                pixel_inds = (f"pixel_{i}", f"bp_{i}")
            elif i == n_sites - 1:
                pixel_shape = (rank_pixel, pixel_dim)
                pixel_inds = (f"bp_{i - 1}", f"pixel_{i}")
            else:
                pixel_shape = (rank_pixel, pixel_dim, rank_pixel)
                pixel_inds = (f"bp_{i - 1}", f"pixel_{i}", f"bp_{i}")

            pixel_node = qt.Tensor(
                torch.randn(*pixel_shape) * init_strength, inds=pixel_inds, tags={f"PixelNode_{i}"}
            )
            tensors.append(pixel_node)

        for i in range(n_sites):
            if i == 0:
                patch_shape = (patch_dim, rank_patch)
                patch_inds = (f"patch_{i}", f"bpa_{i}")
            elif i == n_sites - 1:
                patch_shape = (rank_patch, patch_dim)
                patch_inds = (f"bpa_{i - 1}", f"patch_{i}")
            else:
                patch_shape = (rank_patch, patch_dim, rank_patch)
                patch_inds = (f"bpa_{i - 1}", f"patch_{i}", f"bpa_{i}")

            if i == output_site:
                patch_shape = patch_shape + (output_dim,)
                patch_inds = patch_inds + ("out",)

            patch_node = qt.Tensor(
                torch.randn(*patch_shape) * init_strength, inds=patch_inds, tags={f"PatchNode_{i}"}
            )
            tensors.append(patch_node)

    tn = qt.TensorNetwork(tensors)

    input_labels = [(f"patch_{i}", f"pixel_{i}") for i in range(n_sites)]
    input_dims = [f"site_{i}" for i in range(n_sites)]
    output_dims = ["out"]

    return tn, input_labels, input_dims, output_dims


class CMPO2:
    def __init__(
        self,
        n_sites: int,
        pixel_dim: int,
        patch_dim: int,
        rank_pixel: int,
        rank_patch: int,
        output_dim: int,
        output_site: Optional[int] = None,
        init_strength: float = 0.01,
    ):
        self.n_sites = n_sites
        self.pixel_dim = pixel_dim
        self.patch_dim = patch_dim
        self.rank_pixel = rank_pixel
        self.rank_patch = rank_patch
        self.output_dim = output_dim

        tn, input_labels, input_dims, output_dims = _create_cmpo2(
            n_sites=n_sites,
            pixel_dim=pixel_dim,
            patch_dim=patch_dim,
            rank_pixel=rank_pixel,
            rank_patch=rank_patch,
            output_dim=output_dim,
            output_site=output_site,
            init_strength=init_strength,
        )

        self.tn = tn
        self.input_labels = input_labels
        self.input_dims = input_dims
        self.output_dims = output_dims


class CMPO2_GTN(GTN):
    def __init__(self, cmpo2: CMPO2):
        self.cmpo2 = cmpo2
        self.n_sites = cmpo2.n_sites

        super().__init__(
            tn=cmpo2.tn,
            output_dims=cmpo2.output_dims,
            input_dims=cmpo2.input_dims,
        )

    def construct_nodes(self, x):
        input_nodes = []
        for i in range(self.n_sites):
            node = qt.Tensor(x, inds=["s", f"patch_{i}", f"pixel_{i}"], tags=f"Input_site_{i}")
            input_nodes.append(node)
        return input_nodes


class CMPO2TypeI:
    def __init__(
        self,
        max_sites: int,
        pixel_dim: int,
        patch_dim: int,
        rank_pixel: int,
        rank_patch: int,
        output_dim: int,
        output_site: Optional[int] = None,
        init_strength: float = 0.01,
    ):
        self.max_sites = max_sites
        self.pixel_dim = pixel_dim
        self.patch_dim = patch_dim
        self.rank_pixel = rank_pixel
        self.rank_patch = rank_patch
        self.output_dim = output_dim

        self.tns: List[qt.TensorNetwork] = []
        self.input_dims_list: List[List[str]] = []
        self.input_labels_list: List[List[tuple]] = []
        self.output_dims = ["out"]

        for n_sites in range(1, max_sites + 1):
            tn, input_labels, input_dims, _ = _create_cmpo2(
                n_sites=n_sites,
                pixel_dim=pixel_dim,
                patch_dim=patch_dim,
                rank_pixel=rank_pixel,
                rank_patch=rank_patch,
                output_dim=output_dim,
                output_site=output_site,
                init_strength=init_strength,
            )
            self.tns.append(tn)
            self.input_dims_list.append(input_dims)
            self.input_labels_list.append(input_labels)


class CMPO2TypeI_GTN(torch.nn.Module):
    def __init__(self, cmpo2_typeI: CMPO2TypeI):
        super().__init__()
        self.max_sites = cmpo2_typeI.max_sites
        self.models = torch.nn.ModuleList()

        for i, tn in enumerate(cmpo2_typeI.tns):
            n_sites = i + 1
            gtn = _CMPO2_GTN_Single(tn, n_sites, cmpo2_typeI.output_dims)
            self.models.append(gtn)

    def forward(self, x):
        total = None
        for model in self.models:
            y = model(x)
            if total is None:
                total = y
            else:
                total = total + y
        return total


class _CMPO2_GTN_Single(GTN):
    def __init__(self, tn, n_sites, output_dims):
        self.n_sites = n_sites
        super().__init__(
            tn=tn,
            output_dims=output_dims,
            input_dims=[f"site_{i}" for i in range(n_sites)],
        )

    def construct_nodes(self, x):
        input_nodes = []
        for i in range(self.n_sites):
            node = qt.Tensor(x, inds=["s", f"patch_{i}", f"pixel_{i}"], tags=f"Input_site_{i}")
            input_nodes.append(node)
        return input_nodes


class CMPO2TypeI_NTN:
    """
    NTN Ensemble for CMPO2TypeI with proper 3D input handling.

    This is specialized for CMPO2's (batch, patches, pixels) input format.
    """

    def __init__(
        self,
        cmpo2_typeI: CMPO2TypeI,
        loss,
        X_train: torch.Tensor,
        y_train: torch.Tensor,
        X_val: torch.Tensor = None,
        y_val: torch.Tensor = None,
        batch_size: int = 32,
        method: str = "cholesky",
    ):
        from model.base.NTN import NTN
        from model.base.NTN_Ensemble import NTN_TypeI
        from model.builder import Inputs

        self.max_sites = cmpo2_typeI.max_sites
        self.output_dims = cmpo2_typeI.output_dims
        self.batch_dim = "s"
        self.loss = loss

        self.ntns = []
        self._forward_cache = {}
        self._forward_cache_val = {}

        for idx, (tn, input_labels, input_dims) in enumerate(
            zip(cmpo2_typeI.tns, cmpo2_typeI.input_labels_list, cmpo2_typeI.input_dims_list)
        ):
            train_loader = Inputs(
                inputs=[X_train],
                outputs=[y_train],
                outputs_labels=self.output_dims,
                input_labels=input_labels,
                batch_dim=self.batch_dim,
                batch_size=batch_size,
            )

            all_input_dims = []
            for patch_label, pixel_label in input_labels:
                all_input_dims.extend([patch_label, pixel_label])

            ntn = NTN_TypeI(
                get_others_cached_output_fn=self._get_others_cached_output,
                ntn_index=idx,
                tn=tn,
                output_dims=self.output_dims,
                input_dims=all_input_dims,
                loss=loss,
                data_stream=train_loader,
                method=method,
            )

            if X_val is not None and y_val is not None:
                val_loader = Inputs(
                    inputs=[X_val],
                    outputs=[y_val],
                    outputs_labels=self.output_dims,
                    input_labels=input_labels,
                    batch_dim=self.batch_dim,
                    batch_size=batch_size,
                )
                ntn.val_data = val_loader

            self.ntns.append(ntn)

        self.train_data = self.ntns[0].train_data
        self.val_data = self.ntns[0].val_data if X_val is not None else None

    def _cache_forwards_for_data(self, data_attr, cache_dict):
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

    def _invalidate_cache(self, ntn_idx):
        ntn = self.ntns[ntn_idx]
        batch_outputs = []
        for inputs in ntn.data.data_mu:
            y_pred = ntn._batch_forward(inputs, ntn.tn, [self.batch_dim] + self.output_dims)
            batch_outputs.append(y_pred)
        self._forward_cache[ntn_idx] = batch_outputs

    def _get_others_cached_output(self, exclude_idx, batch_idx):
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

    def _get_all_trainable_nodes(self):
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

    def evaluate(self, metrics, split="train"):
        if split == "train":
            self._cache_all_forwards()
            cache_dict = self._forward_cache
            data_attr = "data"
        elif split == "val":
            self._cache_val_forwards()
            cache_dict = self._forward_cache_val
            data_attr = "val_data"
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
        n_epochs=1,
        regularize=True,
        jitter=1e-6,
        verbose=True,
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
            print(f"Starting CMPO2TypeI NTN Fit: {n_epochs} epochs")
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
