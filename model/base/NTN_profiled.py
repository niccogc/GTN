# type: ignore
from typing import List, Dict, Optional, Tuple
from model.builder import Inputs
from model.utils import print_metrics, REGRESSION_METRICS
from model.exceptions import SingularMatrixError
import torch
import torch.nn as nn
import quimb.tensor as qt
import importlib
import numpy as np
from contextlib import contextmanager

NOT_TRAINABLE_TAG = "NT"
PROFILE = True


@contextmanager
def profile_region(name):
    if not PROFILE or not torch.cuda.is_available():
        yield
        return
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    mem_before = torch.cuda.memory_allocated()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    yield
    end.record()
    torch.cuda.synchronize()
    mem_after = torch.cuda.memory_allocated()
    peak = torch.cuda.max_memory_allocated()
    ms = start.elapsed_time(end)
    print(
        f"[PROFILE] {name}: {ms:.1f}ms | mem: {mem_before / 1024**2:.1f}->{mem_after / 1024**2:.1f}MB | peak: {peak / 1024**2:.1f}MB"
    )


class NTN:
    def __init__(
        self,
        tn,
        output_dims,
        input_dims,
        loss,
        data_stream: Inputs,
        method="cholesky",
        not_trainable_nodes: List[str] = [],
    ):
        super().__init__()

        self.method = method
        self.mse = None
        self.output_dimensions = data_stream.outputs_labels
        self.batch_dim = data_stream.batch_dim
        self.input_indices = data_stream.input_labels
        self.data = data_stream
        self.train_data = data_stream
        self.val_data = None
        self.test_data = None

        self.tn = tn
        not_trainable_nodes = not_trainable_nodes or []
        for node_tag in not_trainable_nodes:
            tensors = self.tn.select_tensors(node_tag, which="any")
            for tensor in tensors:
                tensor.add_tag(NOT_TRAINABLE_TAG)
        self.not_trainable_nodes = not_trainable_nodes
        self.backend = tn.backend
        self.input_dims = input_dims
        self.output_dims = output_dims
        self.loss = loss
        self.singular_encountered = False

    def _concat_over_batches(self, batch_operation, data_iterator, *args, **kwargs):
        """
        Iterates over data_iterator, collects results, and concatenates them.
        """
        results = []

        for batch_idx, batch_data in enumerate(data_iterator):
            inputs = batch_data if isinstance(batch_data, tuple) else (batch_data,)

            batch_res = batch_operation(*inputs, *args, **kwargs)
            results.append(batch_res)

        return self._concat_batch_results(results)

    def _concat_tuple_over_batches(self, batch_operation, data_iterator, *args, **kwargs):
        """
        Specific iterator for operations that return a TUPLE of tensors (e.g. Grad, Hess).
        Concatenates each element of the tuple separately.
        """
        results_list_0 = []
        results_list_1 = []

        for batch_idx, batch_data in enumerate(data_iterator):
            inputs = batch_data if isinstance(batch_data, tuple) else (batch_data,)

            res_0, res_1 = batch_operation(*inputs, *args, **kwargs)

            results_list_0.append(res_0)
            results_list_1.append(res_1)

        return (
            self._concat_batch_results(results_list_0),
            self._concat_batch_results(results_list_1),
        )

    def _sum_over_batches(
        self, batch_operation, data_iterator, *args, **kwargs
    ) -> qt.Tensor | Tuple[qt.Tensor] | None:
        """
        Iterates and sums results on the fly.
        Handles both single Tensors and Tuples of Tensors (e.g. Grad, Hess).
        """
        result = None

        for batch_data in data_iterator:
            inputs = batch_data if isinstance(batch_data, tuple) else (batch_data,)

            batch_result = batch_operation(*inputs, *args, **kwargs)

            if result is None:
                result = batch_result
            else:
                if isinstance(result, tuple) and isinstance(batch_result, tuple):
                    result = tuple(r + b for r, b in zip(result, batch_result))
                else:
                    result = result + batch_result

        return result

    def compute_node_derivatives(
        self, tn: qt.TensorNetwork, node_tag: str, input_generator, sum_over_batches=True
    ):
        """
        Computes the Gradient (Jacobian) and Hessian (approx) for a specific NODE directly.

        Logic per batch:
            1. Calc Environment E (batch, out, node_bonds)
            2. Forward pass (cheaply via E) -> y_pred
            3. Loss Derivs -> dL (batch, out), d2L (batch, out)
            4. Node Jacobian = E * dL
            5. Node Hessian  = (E * d2L) * E^T
        """

        if sum_over_batches:
            return self._sum_over_batches(
                self._batch_node_derivatives, input_generator, node_tag=node_tag
            )
        else:
            return self._concat_tuple_over_batches(
                self._batch_node_derivatives, input_generator, node_tag=node_tag
            )

    def _batch_node_derivatives(self, inputs, y_true, node_tag):
        """
        Worker for a single batch: Returns (Node_Grad, Node_Hess)
        """
        tn = self.tn

        with profile_region(f"env_{node_tag}"):
            env = self._batch_environment(
                inputs, tn, target_tag=node_tag, sum_over_batch=False, sum_over_output=False
            )
        target_tensor = tn[node_tag]

        with profile_region(f"forward_{node_tag}"):
            y_pred = self.forward_from_environment(
                env, node_tag=node_tag, node_tensor=target_tensor, sum_over_batch=False
            )

        with profile_region(f"loss_derivs_{node_tag}"):
            dL_dy, d2L_dy2 = self.loss.get_derivatives(
                y_pred,
                y_true,
                backend=self.backend,
                batch_dim=self.batch_dim,
                output_dims=self.output_dimensions,
                return_hessian_diagonal=False,
            )

        with profile_region(f"grad_contract_{node_tag}"):
            grad_tn = env & dL_dy
            node_inds = target_tensor.inds
            node_grad = grad_tn.contract(output_inds=node_inds)

        out_inds = self.output_dimensions
        out_row_inds = out_inds
        out_col_inds = [x + "_prime" for x in out_inds]

        d2L_tensor = qt.Tensor(d2L_dy2.data, inds=[self.batch_dim] + out_row_inds + out_col_inds)
        env_right = self._prime_indices_tensor(env, exclude_indices=[self.batch_dim])

        with profile_region(f"hess_tn_build_{node_tag}"):
            hess_tn = env & d2L_tensor & env_right

        node_inds = target_tensor.inds
        hess_out_inds = list(node_inds) + [f"{x}_prime" for x in node_inds]

        with profile_region(f"hess_contract_{node_tag}"):
            node_hess = hess_tn.contract(output_inds=hess_out_inds)

        return node_grad, node_hess

    def _compute_H_b(self, node_tag):
        """
        High-level API to get Jacobian and Hessian for a node using the stored data stream.
        """
        J, H = self.compute_node_derivatives(
            self.tn, node_tag, self.data.data_mu_y, sum_over_batches=True
        )

        return J, H

    def _batch_get_derivatives(self, inputs: List[qt.Tensor], y_true, tn):
        """Internal worker: Runs forward pass for ONE batch, then calculates derivatives."""
        output_inds = [self.batch_dim] + self.output_dimensions
        y_pred = self._batch_forward(inputs, tn, output_inds)
        grad, hess = self.loss.get_derivatives(
            y_pred,
            y_true,
            backend=self.backend,
            batch_dim=self.batch_dim,
            output_dims=self.output_dimensions,
            return_hessian_diagonal=self.loss.use_diagonal_hessian,
        )
        return grad, hess

    def compute_derivatives_over_dataset(self, tn: qt.TensorNetwork, input_generator):
        total_grad, total_hess = self._concat_tuple_over_batches(
            self._batch_get_derivatives, input_generator, tn=tn
        )
        return total_grad, total_hess

    def _batch_forward(self, inputs: List[qt.Tensor], tn, output_inds: List[str]) -> qt.Tensor:
        full_tn = tn & inputs
        res = full_tn.contract(output_inds=output_inds)
        if len(output_inds) > 0:
            res.transpose_(*output_inds)
        return res

    def forward(
        self,
        tn: qt.TensorNetwork,
        input_generator,
        sum_over_batch: bool = False,
        sum_over_output: bool = False,
    ):
        if sum_over_output:
            if sum_over_batch:
                target_inds = []
            else:
                target_inds = [self.batch_dim]
        elif sum_over_batch:
            target_inds = self.output_dimensions
        else:
            target_inds = [self.batch_dim] + self.output_dimensions

        if sum_over_batch:
            result = self._sum_over_batches(
                self._batch_forward, input_generator, tn=tn, output_inds=target_inds
            )
        else:
            result = self._concat_over_batches(
                self._batch_forward, input_generator, tn=tn, output_inds=target_inds
            )
        return result

    def forward_from_environment(
        self,
        env: qt.Tensor,
        node_tag: str = None,
        node_tensor: Optional[qt.Tensor] = None,
        sum_over_batch: bool = False,
    ) -> qt.Tensor:
        """
        Compute forward pass using a pre-computed environment.

        This is the KEY computational advantage: once the environment is computed,
        we can quickly evaluate predictions with different node values without
        re-contracting the entire network.

        Args:
            env: Pre-computed environment (from get_environment or _batch_environment)
            node_tag: Tag of the node that was excluded from environment
            node_tensor: Optional custom node tensor. If None, uses current node from self.tn
            sum_over_batch: Whether to sum over batch dimension in output

        Returns:
            y_pred: Predictions with shape [batch, output] or [output] if sum_over_batch

        Example:
            # Compute environment once
            env = model.get_environment(model.tn, 'Node2', input_gen,
                                       sum_over_batch=False, sum_over_output=False)

            # Fast forward passes with different node values
            y_pred_1 = model.forward_from_environment(env, 'Node2', node_tensor=new_value_1)
            y_pred_2 = model.forward_from_environment(env, 'Node2', node_tensor=new_value_2)
            # Much faster than calling forward() twice!
        """
        if node_tensor is None:
            node_tensor = self.tn[node_tag]

        if sum_over_batch:
            output_inds = self.output_dimensions
        else:
            output_inds = [self.batch_dim] + self.output_dimensions

        y_pred = (env & node_tensor).contract(output_inds=output_inds)

        if len(output_inds) > 0:
            y_pred.transpose_(*output_inds)

        return y_pred

    def get_environment(
        self,
        tn: qt.TensorNetwork,
        target_tag: str,
        input_generator,
        copy: bool = True,
        sum_over_batch: bool = False,
        sum_over_output: bool = False,
    ):
        if copy:
            tn_base = tn.copy()
        else:
            tn_base = tn
        if sum_over_batch:
            result = self._sum_over_batches(
                self._batch_environment,
                input_generator,
                tn=tn_base,
                target_tag=target_tag,
                sum_over_batch=sum_over_batch,
                sum_over_output=sum_over_output,
            )
        else:
            result = self._concat_over_batches(
                self._batch_environment,
                input_generator,
                tn=tn_base,
                target_tag=target_tag,
                sum_over_batch=sum_over_batch,
                sum_over_output=sum_over_output,
            )
        return result

    def _batch_environment(
        self,
        inputs,
        tn: qt.TensorNetwork,
        target_tag: str,
        sum_over_batch: bool = False,
        sum_over_output: bool = False,
    ) -> qt.Tensor:
        env_tn = tn & inputs
        env_tn.delete(target_tag)

        outer_inds = list(env_tn.outer_inds())
        final_env_inds = outer_inds.copy()

        if sum_over_batch and self.batch_dim in final_env_inds:
            final_env_inds.remove(self.batch_dim)

        if sum_over_output:
            for out_dim in self.output_dimensions:
                if out_dim in final_env_inds:
                    final_env_inds.remove(out_dim)

        if not sum_over_batch and self.batch_dim not in final_env_inds:
            all_inds_in_env = set()
            for tensor in env_tn:
                all_inds_in_env.update(tensor.inds)

            if self.batch_dim in all_inds_in_env:
                final_env_inds = [self.batch_dim] + final_env_inds

        env_tensor = env_tn.contract(output_inds=final_env_inds)
        return env_tensor

    def get_backend(self, data):
        module = type(data).__module__
        if "torch" in module:
            return "torch", importlib.import_module("torch")
        elif "jax" in module:
            return "jax", importlib.import_module("jax.numpy")
        elif "numpy":
            return "numpy", np

    def outer_operation(self, input_generator, tn, node_tag, sum_over_batches):
        if sum_over_batches:
            result = self._sum_over_batches(
                self._batch_outer_operation,
                input_generator,
                tn=tn,
                node_tag=node_tag,
                sum_over_batches=sum_over_batches,
            )
        else:
            result = self._concat_over_batches(
                self._batch_outer_operation,
                input_generator,
                tn=tn,
                node_tag=node_tag,
                sum_over_batches=sum_over_batches,
            )
        return result

    def _batch_outer_operation(self, inputs, tn, node_tag, sum_over_batches: bool):
        env = self._batch_environment(
            inputs, tn, target_tag=node_tag, sum_over_batch=False, sum_over_output=True
        )

        sample_dim = [self.batch_dim] if not sum_over_batches else []
        env_prime = self._prime_indices_tensor(
            env, exclude_indices=self.output_dimensions + [self.batch_dim]
        )

        env_inds = env.inds + env_prime.inds
        outer_tn = env & env_prime
        out_indices = sample_dim + [i for i in env_inds if i not in [self.batch_dim]]
        batch_result = outer_tn.contract(output_inds=out_indices)
        return batch_result

    def _concat_batch_results(self, batch_results: List):
        if len(batch_results) == 0:
            raise ValueError("No batch results to concatenate")

        first_result = batch_results[0]
        if not isinstance(first_result, qt.Tensor):
            return batch_results

        first_data = first_result.data

        is_torch = False
        try:
            import torch

            if torch.is_tensor(first_data):
                is_torch = True
        except ImportError:
            pass

        if is_torch:
            import torch

            concat_data = torch.cat([t.data for t in batch_results], dim=0)

        elif hasattr(first_data, "__array__"):
            import numpy as np

            concat_data = np.concatenate([t.data for t in batch_results], axis=0)
        else:
            try:
                import numpy as np

                concat_data = np.concatenate([t.data for t in batch_results], axis=0)
            except:
                raise NotImplementedError(
                    f"Concatenation not implemented for backend: {type(first_data)}"
                )

        return qt.Tensor(concat_data, inds=first_result.inds)

    def _to_torch(self, tensor, requires_grad=False):
        import torch

        data = tensor.data if isinstance(tensor, qt.Tensor) else tensor
        if not torch.is_tensor(data):
            data = torch.from_numpy(data)
        if not data.is_floating_point():
            data = data.float()
        if requires_grad:
            data.requires_grad_(True)
        return data

    def _compute_weight_norm_squared(self):
        """
        Compute the squared Frobenius norm of the entire tensor network.
        This is used for L2 regularization: ||weights||^2 = ||TN||_F^2
        """
        norm_sq = self.tn.norm(squared=True)
        if hasattr(norm_sq, "item"):
            norm_sq = norm_sq.item()
        return norm_sq

    def _prime_indices_tensor(
        self,
        tensor: qt.Tensor,
        exclude_indices: Optional[List[str]] = [],
        prime_suffix: str = "_prime",
    ) -> qt.Tensor | None:
        exclude_indices = exclude_indices
        reindex_map = {
            ind: f"{ind}{prime_suffix}" for ind in tensor.inds if ind not in exclude_indices
        }
        return tensor.reindex(reindex_map)

    def cholesky_solve_helper(self, matrix, vector, backend_name, lib):
        """
        Solves the linear system Ax = b using Cholesky decomposition.
        A = L L^T
        1. Solve L y = b
        2. Solve L^T x = y
        """
        if backend_name == "torch":
            b = vector
            if b.ndim == matrix.ndim - 1:
                b = b.unsqueeze(-1)

            try:
                L = lib.linalg.cholesky(matrix)
                x = lib.cholesky_solve(b, L)
                return x.squeeze(-1) if vector.ndim == matrix.ndim - 1 else x
            except RuntimeError:
                return (
                    lib.linalg.solve(matrix, b).squeeze(-1)
                    if vector.ndim == matrix.ndim - 1
                    else lib.linalg.solve(matrix, b)
                )

        elif backend_name == "jax":
            try:
                L = lib.linalg.cholesky(matrix)
                y = lib.linalg.solve(L, vector)
                x = lib.linalg.solve(lib.swapaxes(L, -1, -2).conj(), y)
                return x
            except:
                return lib.linalg.solve(matrix, vector)

        elif backend_name == "numpy":

            def solve_single(A, b):
                try:
                    L = lib.linalg.cholesky(A)
                    y = lib.linalg.solve(L, b)
                    return lib.linalg.solve(L.T, y)
                except lib.linalg.LinAlgError:
                    return lib.linalg.solve(A, b)

            if matrix.ndim == 2:
                return solve_single(matrix, vector)
            else:
                out = []
                for A_i, b_i in zip(matrix, vector):
                    out.append(solve_single(A_i, b_i))
                return lib.array(out)

        else:
            raise ValueError(f"Unknown backend '{backend_name}'.")

    def solve_linear_system(self, matrix_data, vector_data, method="cholesky"):
        """
        Solves Ax = b.

        Args:
            matrix_data: Tensor for A (Batch, N, N)
            vector_data: Tensor for b (Batch, N)
            method: 'cholesky' or 'standard'

        Returns:
            x: Solution tensor
        """
        backend_name, lib = self.get_backend(matrix_data)

        if method == "cholesky":
            result_data = self.cholesky_solve_helper(matrix_data, vector_data, backend_name, lib)
        else:
            if backend_name == "torch":
                b = vector_data
                if b.ndim == matrix_data.ndim - 1:
                    b = b.unsqueeze(-1)
                res = lib.linalg.solve(matrix_data, b)
                result_data = res.squeeze(-1) if vector_data.ndim == matrix_data.ndim - 1 else res
            elif backend_name == "numpy":
                result_data = lib.linalg.solve(matrix_data, vector_data)
            elif backend_name == "jax":
                result_data = lib.linalg.solve(matrix_data, vector_data)

        return result_data

    def _get_node_update(
        self, node_tag, regularize=True, jitter=1e-6, adaptive_jitter=False, max_jitter=0.1
    ):
        with profile_region(f"compute_H_b_{node_tag}"):
            b, H = self._compute_H_b(node_tag)

        variational_ind = b.inds
        map_H = {"rows": variational_ind, "cols": [i + "_prime" for i in variational_ind]}
        map_b = {"cols": variational_ind}

        var_sizes = tuple(self.tn[node_tag].ind_size(i) for i in variational_ind)
        shape_map = {"cols": var_sizes}

        with profile_region(f"fuse_{node_tag}"):
            H.fuse(map_H, inplace=True)
            b.fuse(map_b, inplace=True)

        with profile_region(f"to_dense_{node_tag}"):
            matrix_data = H.to_dense(["rows"], ["cols"])
            gradient_vector = b.to_dense(["cols"])

        backend, lib = self.get_backend(matrix_data)
        if backend == "torch":
            scale = matrix_data.diagonal().abs().mean()
            if not torch.isfinite(scale) or scale == 0:
                scale = torch.tensor(1.0, dtype=matrix_data.dtype)
        elif backend == "numpy":
            scale = lib.abs(lib.diag(matrix_data)).mean()
            if not lib.isfinite(scale) or scale == 0:
                scale = 1.0
        else:
            scale = 1.0

        effective_jitter = jitter
        if adaptive_jitter and regularize:
            backend, lib = self.get_backend(matrix_data)

            try:
                if backend == "torch":
                    eigs = lib.linalg.eigvalsh(matrix_data)
                    min_eig = eigs.min().item()
                    max_eig = eigs.max().item()
                elif backend == "numpy":
                    eigs = lib.linalg.eigvalsh(matrix_data)
                    min_eig = eigs.min()
                    max_eig = eigs.max()
                else:
                    min_eig = None
                    max_eig = None

                if min_eig is not None and max_eig is not None:
                    if min_eig <= 0:
                        effective_jitter = max(effective_jitter, abs(min_eig) * 1.1)

                    if max_eig > 0 and min_eig > 0:
                        cond_number = max_eig / min_eig
                        if cond_number > 1e10:
                            effective_jitter = min(
                                max_jitter, max(effective_jitter, jitter * (cond_number / 1e10))
                            )

                    effective_jitter = min(effective_jitter, max_jitter)
            except:
                pass

        if regularize:
            backend, lib = self.get_backend(matrix_data)

            current_node = self.tn[node_tag].copy()
            current_node.fuse(map_b, inplace=True)
            old_weight = current_node.to_dense(["cols"])

            scaled_jitter = 2 * effective_jitter * scale

            if backend == "torch":
                matrix_data.diagonal().add_(scaled_jitter)
                gradient_vector = gradient_vector + scaled_jitter * old_weight
            elif backend == "numpy":
                rows, cols = lib.diag_indices_from(matrix_data)
                matrix_data[rows, cols] += scaled_jitter
                gradient_vector = gradient_vector + scaled_jitter * old_weight
            elif backend == "jax":
                d_idx = lib.arange(matrix_data.shape[0])
                matrix_data = matrix_data.at[d_idx, d_idx].add(scaled_jitter)
                gradient_vector = gradient_vector + scaled_jitter * old_weight

        with profile_region(f"solve_{node_tag}"):
            tensor_node_data = self.solve_linear_system(matrix_data, -gradient_vector)

        update_node = qt.Tensor(tensor_node_data, inds=["cols"], tags=self.tn[node_tag].tags)

        update_node.unfuse({"cols": variational_ind}, shape_map=shape_map, inplace=True)
        update_node.modify(tags=[node_tag])
        return update_node

    def update_node(self, tensor, node_tag):
        """
        Updates the tensor network by replacing the node with the given tag
        with the new tensor.
        """

        if node_tag in self.tn.tag_map:
            self.tn.delete(node_tag)

        self.tn.add_tensor(tensor)

    def _get_trainable_nodes(self) -> List[str]:
        """
        Identifies all nodes (tags) that should be optimized.
        Excludes nodes with NOT_TRAINABLE_TAG.
        """
        trainable_tags = []
        for tensor in self.tn:
            if NOT_TRAINABLE_TAG in tensor.tags:
                continue
            valid_tags = list(tensor.tags)
            if valid_tags:
                tag = valid_tags[0]
                trainable_tags.append(tag)

        return trainable_tags

    def _batch_evaluate(self, inputs, y_true, tn, metrics):
        """
        Runs forward pass once and applies user metric functions.
        DEBUG version enabled.
        """
        import torch

        with torch.no_grad():
            output_inds = [self.batch_dim] + self.output_dimensions
            y_pred = self._batch_forward(inputs, tn, output_inds)

            y_pred_th = self._to_torch(y_pred)
            y_true_th = self._to_torch(y_true)
            if y_pred_th.numel() == y_true_th.numel():
                if y_pred_th.shape != y_true_th.shape:
                    y_pred_th = y_pred_th.view_as(y_true_th)

            results = {}
            for name, func in metrics.items():
                try:
                    val = func(y_pred_th, y_true_th)
                    results[name] = val
                except Exception as e:
                    print(f"Error in metric '{name}': {e}")
                    print(f"Shapes -> Pred: {y_pred_th.shape}, True: {y_true_th.shape}")
                    raise e

        return results

    def evaluate(self, metrics: Dict[str, callable], data_stream=None, verbose=False):
        """
        Iterates over the dataset and aggregates results from user metrics.

        Args:
            metrics: Dict mapping name -> callable(y_pred, y_true).
                     The callable should return a summable result (e.g. (loss_sum, count)).
            data_stream: Optional Inputs object. If None, uses self.train_data.
        """
        if data_stream is None:
            data_stream = self.train_data

        aggregates = {}

        for i, batch_data in enumerate(data_stream.data_mu_y):
            inputs, y_true = batch_data

            batch_results = self._batch_evaluate(inputs, y_true, self.tn, metrics)

            if i == 0:
                aggregates = batch_results
            else:
                for name, res in batch_results.items():
                    if isinstance(res, tuple):
                        aggregates[name] = tuple(a + b for a, b in zip(aggregates[name], res))
                    else:
                        aggregates[name] += res

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

        if verbose:
            print(f"Evaluation: {final_scores}")

        return final_scores

    def update_tn_node(self, node_tag, regularize, jitter, adaptive_jitter=False, max_jitter=None):
        if max_jitter is None:
            max_jitter = max(10.0, jitter * 10)

        delta_tensor = self._get_node_update(
            node_tag,
            regularize=regularize,
            jitter=jitter,
            adaptive_jitter=adaptive_jitter,
            max_jitter=max_jitter,
        )

        current_tensor = self.tn[node_tag]

        new_tensor = current_tensor + delta_tensor

        self.update_node(new_tensor, node_tag)

    def fit(
        self,
        n_epochs=1,
        regularize=True,
        jitter=1e-6,
        verbose=True,
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
        Main training loop (Alternating Least Squares / Newton Sweep).

        Args:
            eval_metrics: Dict of metric functions. Defaults to REGRESSION_METRICS if None.
            val_data: Optional validation Inputs object for best model selection.
                     If None, uses training data for both optimization and validation.
            test_data: Optional test Inputs object (stored but not used in fit).
            callback_epoch: Optional callback(epoch, scores, info) called after each epoch
            callback_init: Optional callback(scores, info) called after initialization
            patience: Number of epochs to wait for improvement before early stopping (None = no early stopping)
            min_delta: Minimum improvement in quality to be considered as improvement (default: 0.0)
            adaptive_jitter: If True, automatically increase jitter when Hessian is ill-conditioned (default: False)

        Callbacks receive:
            - epoch: int (current epoch number, 0-indexed)
            - scores: dict of metrics from evaluate()
            - info: dict with additional info (jitter, best_score, etc.)

        Best model selection:
            - Uses 'quality' metric (higher is better) on validation data
            - For regression: quality = R²
            - For classification: quality = accuracy
            - Ignores NaN/inf values

        Early stopping with patience:
            - If patience is set, stops training after patience epochs without improvement
            - Improvement is defined as: new_quality > best_quality + min_delta
        """
        self.val_data = val_data
        self.test_data = test_data

        if eval_metrics is None:
            from model.utils import REGRESSION_METRICS

            eval_metrics = REGRESSION_METRICS

        if not isinstance(jitter, list):
            jitter = [jitter] * n_epochs
        trainable_nodes = self._get_trainable_nodes()

        back_sweep = trainable_nodes[-2:0:-1]
        full_sweep_order = trainable_nodes + back_sweep

        if verbose:
            print(f"Starting Fit: {n_epochs} epochs.")
            print(f"Sweep Order: {full_sweep_order}")
            if self.val_data is not None:
                print(f"Validation: Using separate validation set for best model selection")
            if patience is not None:
                print(f"Early stopping: patience={patience}, min_delta={min_delta}")

        scores_train = self.evaluate(eval_metrics, data_stream=self.train_data)

        if self.val_data is not None:
            scores_val = self.evaluate(eval_metrics, data_stream=self.val_data)
        else:
            scores_val = scores_train

        if verbose:
            from model.utils import print_metrics, compute_quality

            print(f"Init    | Train: ", end="")
            print_metrics(scores_train)
            if self.val_data is not None:
                print(f"        | Val:   ", end="")
                print_metrics(scores_val)

        if callback_init is not None:
            info = {
                "n_epochs": n_epochs,
                "jitter_schedule": jitter if isinstance(jitter, list) else [jitter] * n_epochs,
                "regularize": regularize,
            }
            callback_init(scores_train, scores_val, info)

        from model.utils import compute_quality

        best_val_quality = compute_quality(scores_val)
        best_train_quality = compute_quality(scores_train)
        best_scores_train = scores_train.copy()
        best_scores_val = scores_val.copy()
        best_epoch = -1

        patience_counter = 0

        for epoch in range(n_epochs):
            try:
                for node_tag in full_sweep_order:
                    self.update_tn_node(
                        node_tag, regularize, jitter[epoch], adaptive_jitter=adaptive_jitter
                    )

                scores_train = self.evaluate(eval_metrics, data_stream=self.train_data)

                if self.val_data is not None:
                    scores_val = self.evaluate(eval_metrics, data_stream=self.val_data)
                else:
                    scores_val = scores_train

                current_val_quality = compute_quality(scores_val)
                current_train_quality = compute_quality(scores_train)

                current_data_loss = scores_train["loss"]
                current_reg_loss = current_data_loss
                weight_norm_sq = None
                if regularize and jitter[epoch] > 0:
                    weight_norm_sq = self._compute_weight_norm_squared()
                    current_reg_loss += jitter[epoch] * weight_norm_sq

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
                    from model.utils import print_metrics

                    print_metrics(scores_train)
                    if self.val_data is not None:
                        print(f"        | Val:   ", end="")
                        print_metrics(scores_val)
                        print(f"{marker}", end="")
                    print()

                if callback_epoch is not None:
                    info = {
                        "epoch": epoch,
                        "jitter": jitter[epoch],
                        "regularize": regularize,
                        "reg_loss": current_reg_loss,
                        "weight_norm_sq": weight_norm_sq,
                        "best_quality": best_val_quality,
                        "is_best": is_best,
                        "patience_counter": patience_counter,
                    }
                    callback_epoch(epoch, scores_train, scores_val, info)

                if patience is not None and patience_counter >= patience:
                    if verbose:
                        print(
                            f"\n⏸ Early stopping at epoch {epoch + 1} (best was epoch {best_epoch + 1})"
                        )
                    return best_scores_train, best_scores_val

            except torch.linalg.LinAlgError as e:
                self.singular_encountered = True
                if verbose:
                    print(f"\n✗ Singular matrix at epoch {epoch + 1} - stopping training")
                raise SingularMatrixError(
                    message="Singular matrix encountered during NTN optimization", epoch=epoch + 1
                )

        if verbose and best_epoch >= 0:
            print(f"\nBest epoch: {best_epoch + 1} (val_quality={best_val_quality:.6f})")

        return best_scores_train, best_scores_val
