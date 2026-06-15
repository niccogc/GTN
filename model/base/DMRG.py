# type: ignore
import importlib
import math
import os
from typing import List, Dict, Optional, Tuple

import cotengra as ctg
import numpy as np
import quimb.tensor as qt
import torch
import torch.nn as nn

from model.builder import Inputs
from model.exceptions import SingularMatrixError
from model.losses import HuberLoss, MAELoss, MSELoss
from model.utils import REGRESSION_METRICS, compute_quality, print_metrics

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


class DMRG:
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

    def _concat_tuple_over_batches(
        self, batch_operation, data_iterator, *args, **kwargs
    ):
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

    def _sum_over_batches_two_site(
        self, batch_operation, data_iterator, tag_left, tag_right
    ):
        """
        Sum gradient over batches for 2-site gradient-only update.
        
        Returns: (grad_sum, None, fused_inds, bond_idx, left_inds, right_inds)
        Hessian is always None (gradient-only mode).
        """
        grad_sum = None
        fused_inds = None
        bond_idx = None
        left_inds = None
        right_inds = None
        
        for batch_data in data_iterator:
            inputs, y_true = batch_data
            
            grad, hess, f_inds, b_idx, l_inds, r_inds = batch_operation(
                inputs, y_true, tag_left, tag_right
            )
            
            if grad_sum is None:
                grad_sum = grad
                fused_inds = f_inds
                bond_idx = b_idx
                left_inds = l_inds
                right_inds = r_inds
            else:
                grad_sum = grad_sum + grad
        
        return grad_sum, None, fused_inds, bond_idx, left_inds, right_inds

    def compute_node_derivatives(
        self,
        tn: qt.TensorNetwork,
        tag_left: str,
        tag_right: str,
        input_generator,
        sum_over_batches=True,
    ):
        """
        Computes Gradient and Hessian for 2-site fused tensor.
        
        Returns: (grad, hess, fused_inds, bond_idx, left_inds, right_inds)
        """
        if sum_over_batches:
            return self._sum_over_batches_two_site(
                self._batch_node_derivatives, input_generator,
                tag_left=tag_left, tag_right=tag_right
            )
        else:
            raise NotImplementedError("Only sum_over_batches=True supported for 2-site")

    def _batch_node_derivatives(self, inputs, y_true, tag_left, tag_right):
        """
        Compute gradient only for a fused 2-site tensor (gradient-only DMRG).
        
        The fused tensor is formed by contracting tag_left and tag_right over their
        shared bond index. Only the gradient is computed w.r.t. this fused tensor;
        the Hessian is NOT computed (pure gradient descent update).
        """
        tn = self.tn
        
        fused_tensor, bond_idx, left_inds, right_inds = self._fuse_two_site_tensor(
            tag_left, tag_right
        )
        fused_inds = fused_tensor.inds
        
        env = self._batch_environment(
            inputs, tn, tag_left, tag_right,
            sum_over_batch=False, sum_over_output=False
        )
        
        y_pred = self._forward_from_two_site_environment(
            env, fused_tensor, sum_over_batch=False
        )
        
        dL_dy, _ = self.loss.get_derivatives(
            y_pred,
            y_true,
            backend=self.backend,
            batch_dim=self.batch_dim,
            output_dims=self.output_dimensions,
            return_hessian_diagonal=False,
            total_samples=self.train_data.samples,
        )
        
        grad_tn = env & dL_dy
        node_grad = grad_tn.contract(output_inds=fused_inds)
        
        del env, grad_tn, y_pred, dL_dy
        
        return node_grad, None, fused_inds, bond_idx, left_inds, right_inds

    def _compute_H_b(self, tag_left, tag_right):
        """
        Compute gradient and Hessian for 2-site fused tensor using stored data stream.
        """
        result = self.compute_node_derivatives(
            self.tn, tag_left, tag_right, self.data.data_mu_y, sum_over_batches=True
        )
        return result

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
            total_samples=self.train_data.samples,
        )
        return grad, hess

    def compute_derivatives_over_dataset(self, tn: qt.TensorNetwork, input_generator):
        total_grad, total_hess = self._concat_tuple_over_batches(
            self._batch_get_derivatives, input_generator, tn=tn
        )
        return total_grad, total_hess

    def _batch_forward(
        self, inputs: List[qt.Tensor], tn, output_inds: List[str]
    ) -> qt.Tensor:
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

    def _forward_from_two_site_environment(
        self,
        env: qt.Tensor,
        fused_tensor: qt.Tensor,
        sum_over_batch: bool = False,
    ) -> qt.Tensor:
        """
        Compute forward pass using 2-site environment and fused tensor.
        """
        if sum_over_batch:
            output_inds = self.output_dimensions
        else:
            output_inds = [self.batch_dim] + self.output_dimensions
        
        y_pred = (env & fused_tensor).contract(output_inds=output_inds)
        
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
        tag_left: str,
        tag_right: str,
        sum_over_batch: bool = False,
        sum_over_output: bool = False,
    ) -> qt.Tensor:
        """
        Compute environment for 2-site DMRG by excluding TWO adjacent nodes.
        
        The fused tensor AB (from contracting A--bond--B) has indices:
        - External indices of A (excluding bond)
        - External indices of B (excluding bond)
        """
        tensor_left = tn[tag_left]
        tensor_right = tn[tag_right]
        
        bond_idx = self._get_bond_index_between(tag_left, tag_right)
        
        left_inds = set(tensor_left.inds)
        right_inds = set(tensor_right.inds)
        out_labels = set(self.output_dimensions)
        
        # Fused tensor indices = (left_inds ∪ right_inds) - {bond_idx}
        fused_inds = (left_inds | right_inds) - {bond_idx}
        
        env_tn = tn & inputs
        env_tn.delete(tag_left)
        env_tn.delete(tag_right)
        
        # env_inds = {batch_dim} ∪ fused_inds ∪ out_labels - (fused_inds ∩ out_labels)
        intersection = fused_inds & out_labels
        env_inds = ({self.batch_dim} | fused_inds | out_labels) - intersection
        
        if sum_over_batch and self.batch_dim in env_inds:
            env_inds.remove(self.batch_dim)
        
        if sum_over_output:
            for out_dim in self.output_dimensions:
                if out_dim in env_inds:
                    env_inds.remove(out_dim)
        
        env_tensor = env_tn.contract(output_inds=env_inds)
        return env_tensor

    

    def _fuse_two_site_tensor(
        self,
        tag_left: str,
        tag_right: str,
    ) -> Tuple[qt.Tensor, str, Tuple[str, ...], Tuple[str, ...]]:
        """
        Fuse two adjacent tensors into a single 2-site tensor.
        
        Returns:
            fused_tensor: The contracted tensor (without the internal bond)
            bond_idx: The bond index that was contracted
            left_inds: Original indices of left tensor (for unfusing)
            right_inds: Original indices of right tensor (for unfusing)
        """
        tensor_left = self.tn[tag_left]
        tensor_right = self.tn[tag_right]
        
        bond_idx = self._get_bond_index_between(tag_left, tag_right)
        
        left_inds = tuple(tensor_left.inds)
        right_inds = tuple(tensor_right.inds)
        
        # Contract over the shared bond index
        fused_tensor = (tensor_left & tensor_right).contract()
        
        return fused_tensor, bond_idx, left_inds, right_inds

    def _svd_split_two_site_tensor(
        self,
        fused_tensor: qt.Tensor,
        bond_idx: str,
        left_inds: Tuple[str, ...],
        right_inds: Tuple[str, ...],
        max_bond: Optional[int] = None,
        cutoff: float = 1e-10,
        absorb: str = "right",
    ) -> Tuple[qt.Tensor, qt.Tensor, int]:
        """
        Split a fused 2-site tensor back into two tensors using SVD.
        """
        left_external = [i for i in left_inds if i != bond_idx]
        
        new_left, new_right = fused_tensor.split(
            left_inds=left_external,
            right_inds=None,
            method="svd",
            get="tensors",
            bond_ind=bond_idx,
            max_bond=max_bond,
            cutoff=cutoff,
            absorb=absorb,
        )
        
        new_bond_dim = new_left.ind_size(bond_idx)
        
        return new_left, new_right, new_bond_dim

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

        if self.batch_dim in first_result.inds:
            batch_results = [t.moveindex(self.batch_dim, 0) for t in batch_results]
            first_result = batch_results[0]

        first_data = first_result.data

        is_torch = torch.is_tensor(first_data)

        if is_torch:
            concat_data = torch.cat([t.data for t in batch_results], dim=0)

        elif hasattr(first_data, "__array__"):
            concat_data = np.concatenate([t.data for t in batch_results], axis=0)
        else:
            try:
                concat_data = np.concatenate([t.data for t in batch_results], axis=0)
            except Exception:
                raise NotImplementedError(
                    f"Concatenation not implemented for backend: {type(first_data)}"
                )

        return qt.Tensor(concat_data, inds=first_result.inds)

    def _to_torch(self, tensor, requires_grad=False):
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
        This is used for L2 regularization: ||weights||^2 = sum_i ||T_i||_F^2
        For disconnected networks (like CMPO2), we sum norms of individual tensors.
        """
        try:
            norm_sq = self.tn.norm(squared=True)
            if hasattr(norm_sq, "item"):
                norm_sq = norm_sq.item()
            return norm_sq
        except ValueError:
            total = 0.0
            for t in self.tn.tensors:
                t_norm = (t.data ** 2).sum()
                if hasattr(t_norm, "item"):
                    t_norm = t_norm.item()
                total += t_norm
            return total

    def _prime_indices_tensor(
        self,
        tensor: qt.Tensor,
        exclude_indices: Optional[List[str]] = [],
        prime_suffix: str = "_prime",
    ) -> qt.Tensor | None:
        exclude_indices = exclude_indices
        reindex_map = {
            ind: f"{ind}{prime_suffix}"
            for ind in tensor.inds
            if ind not in exclude_indices
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
            result_data = self.cholesky_solve_helper(
                matrix_data, vector_data, backend_name, lib
            )
        else:
            if backend_name == "torch":
                b = vector_data
                if b.ndim == matrix_data.ndim - 1:
                    b = b.unsqueeze(-1)
                res = lib.linalg.solve(matrix_data, b)
                result_data = (
                    res.squeeze(-1) if vector_data.ndim == matrix_data.ndim - 1 else res
                )
            elif backend_name == "numpy":
                result_data = lib.linalg.solve(matrix_data, vector_data)
            elif backend_name == "jax":
                result_data = lib.linalg.solve(matrix_data, vector_data)

        return result_data

    def _get_node_optimum_regression(self, node_tag, regularize=True, jitter=1e-6, **_):
        target = self.tn[node_tag]
        node_inds = set(target.inds)
        out_labels = set(self.output_dimensions)

        env = self.get_environment(
            self.tn,
            node_tag,
            self.train_data.data_mu,
            copy=False,
            sum_over_batch=False,
            sum_over_output=True,
        )
        y = self.train_data.outputs_data[0].squeeze()

        env_inds = set(env.inds)
        variational_inds = list(env_inds & node_inds)
        var_sizes = tuple(target.ind_size(i) for i in variational_inds)

        env.fuse_({self.batch_dim: [self.batch_dim], "cols": variational_inds})
        env.moveindex_(self.batch_dim, 0)
        env_matrix = env.data

        if regularize and jitter > 0:
            n_params = env_matrix.shape[1]
            sqrt_j = np.sqrt(jitter)
            env_matrix = torch.cat(
                [
                    env_matrix,
                    sqrt_j
                    * torch.eye(
                        n_params, dtype=env_matrix.dtype, device=env_matrix.device
                    ),
                ],
                dim=0,
            )
            y = torch.cat(
                [y, torch.zeros(n_params, dtype=y.dtype, device=y.device)], dim=0
            )

        solution = torch.linalg.lstsq(env_matrix, y).solution

        result = qt.Tensor(data=solution, inds=["cols"])
        result.unfuse_({"cols": variational_inds}, shape_map={"cols": var_sizes})

        for out_ind in (node_inds & out_labels) - env_inds:
            result = (result & qt.Tensor(data=torch.ones(1, device = result.data.device, dtype= result.data.dtype), inds=[out_ind])).contract()

        result.modify(tags=[node_tag])
        return result

    def _get_node_update(
        self,
        tag_left,
        tag_right,
        learning_rate=0.01,
        regularize=True,
        jitter=1e-6,
    ):
        """
        Gradient descent update for 2-site fused tensor.
        
        Computes only the gradient (no Hessian), then applies:
            new_fused = old_fused - learning_rate * gradient
        
        Returns the NEW fused tensor (not a delta).
        """
        grad, hess, fused_inds, bond_idx, left_inds, right_inds = self._compute_H_b(
            tag_left, tag_right
        )
        # hess is always None in gradient-only mode
        
        variational_ind = list(fused_inds)
        map_b = {"cols": variational_ind}
        
        fused_tensor, _, _, _ = self._fuse_two_site_tensor(tag_left, tag_right)
        var_sizes = tuple(fused_tensor.ind_size(i) for i in variational_ind)
        shape_map = {"cols": var_sizes}
        
        grad.fuse(map_b, inplace=True)
        gradient_vector = grad.to_dense(["cols"])
        
        current_fused = fused_tensor.copy()
        current_fused.fuse(map_b, inplace=True)
        old_weight = current_fused.to_dense(["cols"])
        
        if regularize:
            # L2 weight decay: add 2*jitter*old_weight to gradient
            gradient_vector = gradient_vector + 2 * jitter * old_weight
        
        # Gradient descent: new = old - lr * g
        new_data = old_weight - learning_rate * gradient_vector
        
        new_fused = qt.Tensor(new_data, inds=["cols"])
        new_fused.unfuse({"cols": variational_ind}, shape_map=shape_map, inplace=True)
        
        del hess, grad, gradient_vector
        
        return new_fused, bond_idx, left_inds, right_inds

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

    def _get_trainable_node_pairs(self) -> List[Tuple[str, str]]:
        """
        Identifies adjacent node pairs for 2-site DMRG sweeping.
        
        For an MPS with nodes [Node0, Node1, Node2, ..., NodeN-1]:
        Returns pairs: [(Node0, Node1), (Node1, Node2), ..., (NodeN-2, NodeN-1)]
        
        Only includes pairs where BOTH nodes are trainable.
        Assumes nodes are tagged as 'Node{i}' and connected via bond index 'b{i}'.
        """
        trainable_tags = set(self._get_trainable_nodes())
        
        # Extract node indices and sort
        node_indices = []
        for tag in trainable_tags:
            if tag.startswith("Node"):
                try:
                    idx = int(tag[4:])
                    node_indices.append(idx)
                except ValueError:
                    continue
        
        node_indices.sort()
        
        # Build adjacent pairs
        pairs = []
        for i in range(len(node_indices) - 1):
            idx_left = node_indices[i]
            idx_right = node_indices[i + 1]
            # Only include if they are actually adjacent (consecutive indices)
            if idx_right == idx_left + 1:
                tag_left = f"Node{idx_left}"
                tag_right = f"Node{idx_right}"
                if tag_left in trainable_tags and tag_right in trainable_tags:
                    pairs.append((tag_left, tag_right))
        
        return pairs

    def _get_bond_index_between(self, tag_left: str, tag_right: str) -> Optional[str]:
        """
        Find the shared bond index between two adjacent nodes.
        
        For MPS structure, Node{i} and Node{i+1} share bond index 'b{i}'.
        
        Returns:
            The shared index name, or None if nodes don't share an index.
        """
        tensor_left = self.tn[tag_left]
        tensor_right = self.tn[tag_right]
        
        shared_inds = set(tensor_left.inds) & set(tensor_right.inds)
        
        if len(shared_inds) == 0:
            return None
        elif len(shared_inds) == 1:
            return shared_inds.pop()
        else:
            # Multiple shared indices - pick the bond index (starts with 'b')
            for ind in shared_inds:
                if ind.startswith('b'):
                    return ind
            return shared_inds.pop()

    def _batch_evaluate(self, inputs, y_true, tn, metrics):
        """
        Runs forward pass once and applies user metric functions.
        DEBUG version enabled.
        """
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
                        aggregates[name] = tuple(
                            a + b for a, b in zip(aggregates[name], res)
                        )
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

    def update_tn_node(
        self,
        tag_left,
        tag_right,
        learning_rate=0.01,
        regularize=True,
        jitter=1e-6,
        max_bond: Optional[int] = None,
        cutoff: float = 1e-10,
        absorb: str = "right",
    ):
        """
        Full 2-site DMRG gradient descent update step:
        1. Compute gradient for fused tensor
        2. Gradient descent: new = old - lr * gradient
        3. SVD split back into two tensors
        4. Update both nodes in the TN
        """
        new_fused, bond_idx, left_inds, right_inds = self._get_node_update(
            tag_left, tag_right,
            learning_rate=learning_rate,
            regularize=regularize,
            jitter=jitter,
        )
        
        new_left, new_right, new_bond_dim = self._svd_split_two_site_tensor(
            new_fused, bond_idx, left_inds, right_inds,
            max_bond=max_bond,
            cutoff=cutoff,
            absorb=absorb,
        )
        
        new_left.modify(tags={tag_left})
        new_right.modify(tags={tag_right})
        
        self.update_node(new_left, tag_left)
        self.update_node(new_right, tag_right)
        
        return new_bond_dim

    def update_tn_node_optimum(self, node_tag, regularize, jitter):
        """
        Update a node by directly computing its optimal value using least squares.

        Unlike update_tn_node which computes a delta and adds it to the current value,
        this method directly solves for the optimal node value.

        This approach is specific to regression tasks with a single output dimension.
        """
        optimal_tensor = self._get_node_optimum_regression(
            node_tag,
            regularize=regularize,
            jitter=jitter,
        )
        self.update_node(optimal_tensor, node_tag)

    def fit(
        self,
        n_epochs=1,
        learning_rate=0.01,
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
        max_bond: Optional[int] = None,
        cutoff: float = 1e-10,
    ):
        """
        2-site DMRG training loop with gradient-only updates.
        
        Sweeps through adjacent node pairs, optimizing the fused 2-site tensor
        with gradient descent and splitting back with SVD.
        """
        self.val_data = val_data
        self.test_data = test_data
        
        if eval_metrics is None:
            eval_metrics = REGRESSION_METRICS
        
        if not isinstance(jitter, list):
            jitter = [jitter] * n_epochs
        
        pairs = self._get_trainable_node_pairs()
        forward_sweep = pairs
        backward_sweep = pairs[::-1]
        
        if verbose:
            print(f"Starting 2-site DMRG (gradient-only): {n_epochs} epochs")
            print(f"Node pairs: {len(pairs)}")
            print(f"Learning rate: {learning_rate}")
            if max_bond:
                print(f"Max bond dimension: {max_bond}")
        
        scores_train = self.evaluate(eval_metrics, data_stream=self.train_data)
        scores_val = self.evaluate(eval_metrics, data_stream=self.val_data) if self.val_data else scores_train
        
        if verbose:
            print(f"Init    | Train: ", end="")
            print_metrics(scores_train)
            if self.val_data:
                print(f"        | Val:   ", end="")
                print_metrics(scores_val)
        
        if callback_init:
            callback_init(scores_train, scores_val, {"n_epochs": n_epochs, "jitter_schedule": jitter})
        
        best_val_quality = compute_quality(scores_val)
        best_scores_train = scores_train.copy()
        best_scores_val = scores_val.copy()
        best_epoch = -1
        patience_counter = 0
        
        for epoch in range(n_epochs):
            try:
                with torch.no_grad():
                    sweep = forward_sweep if epoch % 2 == 0 else backward_sweep
                    absorb = "right" if epoch % 2 == 0 else "left"
                    
                    for tag_left, tag_right in sweep:
                        self.update_tn_node(
                            tag_left, tag_right,
                            learning_rate=learning_rate,
                            regularize=regularize,
                            jitter=jitter[epoch],
                            max_bond=max_bond,
                            cutoff=cutoff,
                            absorb=absorb,
                        )
                
                scores_train = self.evaluate(eval_metrics, data_stream=self.train_data)
                scores_val = self.evaluate(eval_metrics, data_stream=self.val_data) if self.val_data else scores_train
                
                if not math.isfinite(scores_train.get("loss", float("inf"))):
                    raise SingularMatrixError(message="NaN loss", epoch=epoch + 1)
                
                current_val_quality = compute_quality(scores_val)
                
                if current_val_quality is not None and math.isfinite(current_val_quality):
                    if current_val_quality > best_val_quality + min_delta:
                        best_val_quality = current_val_quality
                        best_scores_train = scores_train.copy()
                        best_scores_val = scores_val.copy()
                        best_epoch = epoch
                        is_best = True
                        patience_counter = 0
                    else:
                        is_best = False
                        patience_counter += 1
                else:
                    is_best = False
                    patience_counter += 1
                
                if verbose:
                    marker = " *" if is_best else ""
                    print(f"Epoch {epoch + 1} | Train: ", end="")
                    print_metrics(scores_train)
                    if self.val_data:
                        print(f"        | Val:   ", end="")
                        print_metrics(scores_val)
                    print(marker)
                
                if callback_epoch:
                    callback_epoch(epoch, scores_train, scores_val, {
                        "jitter": jitter[epoch],
                        "best_quality": best_val_quality,
                        "is_best": is_best,
                    })
                
                if patience and patience_counter >= patience:
                    if verbose:
                        print(f"\n⏸ Early stopping at epoch {epoch + 1}")
                    return best_scores_train, best_scores_val
            
            except torch.linalg.LinAlgError:
                self.singular_encountered = True
                raise SingularMatrixError(message="Singular matrix", epoch=epoch + 1)
        
        if verbose and best_epoch >= 0:
            print(f"\nBest epoch: {best_epoch + 1} (quality={best_val_quality:.6f})")
        
        return best_scores_train, best_scores_val
