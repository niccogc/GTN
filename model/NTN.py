# type: ignore
from typing import List, Dict, Optional, Tuple
from model.builder import Inputs
from model.utils import print_metrics, REGRESSION_METRICS
import torch
# from builder import Inputs # Assuming this exists in your project
import torch.nn as nn
import quimb.tensor as qt
import importlib
import numpy as np

NOT_TRAINABLE_TAG = "NT"

class NTN():
    def __init__(self,
                 tn, output_dims, input_dims, loss, data_stream: Inputs,
                 method = 'cholesky',
                 not_trainable_nodes: List[str] = [],
             ):
        super().__init__()
        
        self.method = method
        self.mse = None
        self.output_dimensions = data_stream.outputs_labels
        self.batch_dim = data_stream.batch_dim
        self.input_indices = data_stream.input_labels
        self.data = data_stream
        
        # Tag not trainable nodes with NT tag
        not_trainable_nodes = not_trainable_nodes or []
        for node_tag in not_trainable_nodes:
            # Get tensor(s) with this tag and add NT tag
            tensors = self.tn.select_tensors(node_tag, which='any')
            for tensor in tensors:
                tensor.add_tag(NOT_TRAINABLE_TAG)
        # Store the list of not trainable node tags
        self.not_trainable_nodes = not_trainable_nodes
        self.tn = tn
        self.backend = tn.backend
        self.input_dims = input_dims
        self.output_dims = output_dims
        self.loss = loss

    # -------------------------------------------------------------------------
    # ITERATION HELPERS
    # -------------------------------------------------------------------------

    def _concat_over_batches(self, 
                             batch_operation, 
                             data_iterator, 
                             *args, 
                             **kwargs
                          ):
        """
        Iterates over data_iterator, collects results, and concatenates them.
        """
        results = []
        
        for batch_idx, batch_data in enumerate(data_iterator):
            # Ensure proper unpacking (handle tuple vs single item)
            inputs = batch_data if isinstance(batch_data, tuple) else (batch_data,)
            
            # Execute operation
            batch_res = batch_operation(*inputs, *args, **kwargs)
            results.append(batch_res)
            
        return self._concat_batch_results(results)

    def _concat_tuple_over_batches(self, 
                             batch_operation, 
                             data_iterator, 
                             *args, 
                             **kwargs
                          ):
        """
        Specific iterator for operations that return a TUPLE of tensors (e.g. Grad, Hess).
        Concatenates each element of the tuple separately.
        """
        results_list_0 = []
        results_list_1 = []
        
        for batch_idx, batch_data in enumerate(data_iterator):
            inputs = batch_data if isinstance(batch_data, tuple) else (batch_data,)
            
            # Expecting (res_0, res_1)
            res_0, res_1 = batch_operation(*inputs, *args, **kwargs)
            
            results_list_0.append(res_0)
            results_list_1.append(res_1)
            
        return (
            self._concat_batch_results(results_list_0),
            self._concat_batch_results(results_list_1)
        )

    def _sum_over_batches(self, 
                          batch_operation, 
                          data_iterator, 
                          *args, 
                          **kwargs) -> qt.Tensor | Tuple[qt.Tensor] | None:
        """
        Iterates and sums results on the fly. 
        Handles both single Tensors and Tuples of Tensors (e.g. Grad, Hess).
        """
        result = None
        
        for batch_data in data_iterator:
            # Ensure data is a tuple for unpacking
            inputs = batch_data if isinstance(batch_data, tuple) else (batch_data,)
            
            # Execute operation
            batch_result = batch_operation(*inputs, *args, **kwargs)
            
            if result is None:
                result = batch_result
            else:
                # If result is a tuple (Grad, Hess), sum element-wise
                if isinstance(result, tuple) and isinstance(batch_result, tuple):
                    result = tuple(r + b for r, b in zip(result, batch_result))
                else:
                    # Standard tensor addition
                    result = result + batch_result
            
        return result

    # -------------------------------------------------------------------------
    # DERIVATIVE & OPTIMIZATION LOGIC
    # -------------------------------------------------------------------------

    def compute_node_derivatives(self, 
                                tn: qt.TensorNetwork, 
                                node_tag: str, 
                                input_generator,
                                sum_over_batches=True
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
        
        # Uses the improved _sum_over_batches which now handles tuples correctly
        if sum_over_batches:
            return self._sum_over_batches(
                self._batch_node_derivatives,
                input_generator,
                node_tag=node_tag
            )
        else:
            # Use tuple concat if not summing (rarely needed)
            return self._concat_tuple_over_batches(
                self._batch_node_derivatives,
                input_generator,
                node_tag=node_tag
            )

    def _batch_node_derivatives(self, inputs, y_true, node_tag):
        """
        Worker for a single batch: Returns (Node_Grad, Node_Hess)
        """
        # 1. Calculate Environment E 
        # Indices: [batch_dim, output_dims..., node_open_bonds...]
        tn = self.tn
        env = self._batch_environment(
            inputs, 
            tn, 
            target_tag=node_tag, 
            sum_over_batch=False, 
            sum_over_output=False 
        )
        # 2. Reconstruct y_pred from Environment + Current Node
        # This avoids running a full forward pass separately
        target_tensor = tn[node_tag]
        
        # Contract E with Node to get Prediction
        # y_pred = E @ Node
        y_pred = (env & target_tensor).contract(output_inds=[self.batch_dim] + self.output_dimensions)

        # 3. Compute Loss Derivatives (w.r.t Output)
        # dL_dy: [batch, output]
        # d2L_dy2: [batch, output] (Diagonal approximation)
        # d2L_dy2 is now (Batch, Out, Out) if diagonal=False
        dL_dy, d2L_dy2 = self.get_derivatives(y_pred, y_true, return_hessian_diagonal=False)
        grad_tn = env & dL_dy
        # Result indices should be just the node indices (no batch, no output)
        node_inds = target_tensor.inds
        node_grad = grad_tn.contract(output_inds=node_inds)
        out_inds = self.output_dimensions
        
        out_row_inds = out_inds
        out_col_inds = [x + "_prime" for x in out_inds]
        
        # 1. Cast d2L data to qt.Tensor with indices (Batch, Out_Row, Out_Col)
        # Note: self.output_dimensions usually has length 1 for simple regression
        d2L_tensor = qt.Tensor(
            d2L_dy2.data, 
            inds=[self.batch_dim] + out_row_inds + out_col_inds
        )
        # Prepare Env_Right (Prime bonds, rename output to _col)
        env_right = self._prime_indices_tensor(env, exclude_indices= [self.batch_dim])
        
        # Full Hessian Network
        hess_tn = env & d2L_tensor & env_right
        
        # Output indices: Bonds + Bonds_Prime (sum over Batch, Out_Row, Out_Col)
        node_inds = target_tensor.inds
        hess_out_inds = list(node_inds) + [f"{x}_prime" for x in node_inds]
        
        node_hess = hess_tn.contract(output_inds=hess_out_inds)

        return node_grad, node_hess

    def _compute_H_b(self, node_tag):
        """
        High-level API to get Jacobian and Hessian for a node using the stored data stream.
        """
        J, H = self.compute_node_derivatives(
            self.tn, 
            node_tag, 
            self.data.data_mu_y, 
            sum_over_batches=True
        )

        return J, H

    def get_derivatives(self, y_pred, y_true, return_hessian_diagonal=True):
        import torch
        # 1. Prepare Data
        y_pred_th = self._to_torch(y_pred, requires_grad=True)
        y_true_th = self._to_torch(y_true, requires_grad=False)
        
        if y_pred_th.device != y_true_th.device:
            y_true_th = y_true_th.to(y_pred_th.device)

        batch_sz = y_pred_th.shape[0]
        y_flat = y_pred_th.view(batch_sz, -1)
        num_outputs = y_flat.shape[1]

        # 2. Compute Loss
        loss_val = self.loss(y_pred_th, y_true_th)
        
        # 3. First Derivative
        grad_th = torch.autograd.grad(loss_val, y_pred_th, create_graph=True)[0]
        grad_flat = grad_th.view(batch_sz, -1)
        
        # 4. Second Derivative
        if return_hessian_diagonal:
            hess_cols = []
            for i in range(num_outputs):
                grad_sum = grad_flat[:, i].sum()
                h_i = torch.autograd.grad(grad_sum, y_pred_th, retain_graph=True)[0]
                h_i_flat = h_i.view(batch_sz, -1)
                hess_cols.append(h_i_flat[:, i])
            
            hess_th = torch.stack(hess_cols, dim=1).view(y_pred_th.shape)
            
            # Indices match y_pred
            hess_inds = y_pred.inds if isinstance(y_pred, qt.Tensor) else None

        else:
            # Full Per-Sample Hessian: (Batch, Out, Out)
            hess_rows = []
            for i in range(num_outputs):
                grad_sum = grad_flat[:, i].sum()
                h_row = torch.autograd.grad(grad_sum, y_pred_th, retain_graph=True)[0]
                hess_rows.append(h_row.view(batch_sz, -1))
            
            # (Batch, Out, Out)
            hess_th = torch.stack(hess_rows, dim=0).permute(1, 0, 2)
            
            # Create proper indices for the full matrix [batch, out, out_col]
            if isinstance(y_pred, qt.Tensor):
                out_inds = [i for i in y_pred.inds if i != self.batch_dim]
                out_inds_col = [i + "_col" for i in out_inds]
                hess_inds = [self.batch_dim] + out_inds + out_inds_col
            else:
                hess_inds = None

        # 5. Wrap back
        if self.backend == 'numpy':
            grad_data = grad_th.detach().numpy()
            hess_data = hess_th.detach().numpy()
        else:
            grad_data = grad_th
            hess_data = hess_th
        
        grad_inds = y_pred.inds if isinstance(y_pred, qt.Tensor) else None
        return qt.Tensor(grad_data, inds=grad_inds), qt.Tensor(hess_data, inds=hess_inds)

    def _batch_get_derivatives(self, inputs: List[qt.Tensor], y_true, tn):
        """Internal worker: Runs forward pass for ONE batch, then calculates derivatives."""
        output_inds = [self.batch_dim] + self.output_dimensions
        y_pred = self._batch_forward(inputs, tn, output_inds)
        grad, hess = self.get_derivatives(y_pred, y_true)
        return grad, hess

    def compute_derivatives_over_dataset(self, tn: qt.TensorNetwork, input_generator):
        total_grad, total_hess = self._concat_tuple_over_batches(
            self._batch_get_derivatives,
            input_generator,
            tn=tn
        )
        return total_grad, total_hess

    # -------------------------------------------------------------------------
    # EXISTING CORE HELPERS
    # -------------------------------------------------------------------------

    def _batch_forward(self, inputs: List[qt.Tensor], tn, output_inds: List[str]) -> qt.Tensor:
        full_tn = tn & inputs
        res = full_tn.contract(output_inds=output_inds)
        if len(output_inds) > 0:
            res.transpose_(*output_inds)
        return res 

    def forward(self, tn: qt.TensorNetwork, input_generator, 
                sum_over_batch: bool = False, sum_over_output: bool = False):
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
                self._batch_forward,
                input_generator,
                tn = tn,
                output_inds = target_inds
            )
        else:
            result = self._concat_over_batches(
                self._batch_forward,
                input_generator,
                tn = tn,
                output_inds = target_inds
            )
        return result
    
    def get_environment(self, tn: qt.TensorNetwork,
                                 target_tag: str, 
                                 input_generator,
                                 copy: bool = True,
                                 sum_over_batch: bool = False, 
                                 sum_over_output: bool = False
                             ):
        if copy:
            tn_base = tn.copy()
        else:
            tn_base = tn
        if sum_over_batch:
            result = self._sum_over_batches(
                                self._batch_environment,
                                input_generator,
                                tn = tn_base,
                                target_tag=target_tag,
                                sum_over_batch=sum_over_batch,
                                sum_over_output=sum_over_output
                          )
        else:
            result = self._concat_over_batches(
                                self._batch_environment,
                                input_generator,
                                tn = tn_base,
                                target_tag=target_tag,
                                sum_over_batch=sum_over_batch,
                                sum_over_output=sum_over_output
                          )
        return result

    def _batch_environment(self, inputs, tn: qt.TensorNetwork, target_tag: str,
                        sum_over_batch: bool = False, sum_over_output: bool = False) -> qt.Tensor:
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
        if 'torch' in module:
            return 'torch', importlib.import_module('torch')
        elif 'jax' in module:
            return 'jax', importlib.import_module('jax.numpy')
        elif 'numpy':
            return 'numpy', np

    def outer_operation(self, input_generator, tn, node_tag, sum_over_batches):
        if sum_over_batches:
            result = self._sum_over_batches(
                self._batch_outer_operation,
                input_generator,
                tn = tn,
                node_tag = node_tag,
                sum_over_batches=sum_over_batches
            )
        else:
            result = self._concat_over_batches(
                self._batch_outer_operation,
                input_generator,
                tn = tn,
                node_tag = node_tag,
                sum_over_batches=sum_over_batches
            )
        return result
    
    def _batch_outer_operation(self, inputs, tn, node_tag, sum_over_batches: bool):
        env = self._batch_environment(
            inputs,
            tn,
            target_tag=node_tag,
            sum_over_batch=False,
            sum_over_output=True
        )
        
        sample_dim = [self.batch_dim] if not sum_over_batches else []
        env_prime = self._prime_indices_tensor(env, exclude_indices=self.output_dimensions+[self.batch_dim])

        env_inds = env.inds + env_prime.inds
        outer_tn = env & env_prime
        out_indices = sample_dim + [i for i in env_inds if i not in [self.batch_dim]]
        batch_result = outer_tn.contract(output_inds = out_indices)
        return batch_result

    def _concat_batch_results(self, batch_results: List):
        if len(batch_results) == 0:
            raise ValueError("No batch results to concatenate")
        
        first_result = batch_results[0]
        if not isinstance(first_result, qt.Tensor):
            return batch_results
        
        first_data = first_result.data
        
        # 1. Check for PyTorch Tensors specifically
        is_torch = False
        try:
            import torch
            if torch.is_tensor(first_data):
                is_torch = True
        except ImportError:
            pass

        if is_torch:
            import torch
            # Use torch.cat to preserve gradients and handle tensors correctly
            concat_data = torch.cat([t.data for t in batch_results], dim=0)
            
        # 2. Fallback to NumPy for others
        elif hasattr(first_data, '__array__'):  
            import numpy as np
            concat_data = np.concatenate([t.data for t in batch_results], axis=0)
        else:
            try:
                import numpy as np
                concat_data = np.concatenate([t.data for t in batch_results], axis=0)
            except:
                raise NotImplementedError(f"Concatenation not implemented for backend: {type(first_data)}")
        
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

    def _prime_indices_tensor(self, 
                            tensor: qt.Tensor,
                            exclude_indices: Optional[List[str]] = [],
                            prime_suffix: str = "_prime") -> qt.Tensor | None:
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
        if backend_name == 'torch':
            # vector shape: (Batch, N) -> (Batch, N, 1) for torch.cholesky_solve
            # But torch.linalg.cholesky works on batches naturally
            
            # Ensure vector is at least 2D (column vector) for the solver if strictly 1D provided
            b = vector
            if b.ndim == matrix.ndim - 1:
                b = b.unsqueeze(-1)
            
            try:
                L = lib.linalg.cholesky(matrix)
                # torch.cholesky_solve(b, L) solves Ax=b given L
                x = lib.cholesky_solve(b, L)
                return x.squeeze(-1) if vector.ndim == matrix.ndim - 1 else x
            except RuntimeError:
                # Fallback if not positive definite
                return lib.linalg.solve(matrix, b).squeeze(-1) if vector.ndim == matrix.ndim - 1 else lib.linalg.solve(matrix, b)

        elif backend_name == 'jax':
            # JAX usually has jax.scipy.linalg.cho_solve, but let's stick to jax.numpy
            # Standard solve is robust in JAX, manual Cholesky solve is verbose without scipy
            try:
                L = lib.linalg.cholesky(matrix)
                # Solve L y = b
                y = lib.linalg.solve(L, vector)
                # Solve L.T x = y
                # jax cholesky returns lower triangular
                x = lib.linalg.solve(lib.swapaxes(L, -1, -2).conj(), y)
                return x
            except:
                 return lib.linalg.solve(matrix, vector)

        elif backend_name == 'numpy':
            # Numpy doesn't have a specific cholesky_solve in .linalg (scipy does)
            # We use standard solve which is faster and more stable than inv(A) @ b
            # If cholesky is strictly required for stability on PD matrices:
            
            # Helper for single solve
            def solve_single(A, b):
                try:
                    L = lib.linalg.cholesky(A)
                    # Forward substitution: L y = b
                    # Since numpy doesn't have solve_triangular in main linalg, we might just use standard solve
                    # Standard solve detects structure often, or we just trust lapack.
                    # Ideally: use scipy.linalg.cho_solve if available, else standard solve.
                    return lib.linalg.solve(A, b)
                except lib.linalg.LinAlgError:
                    # Fallback to least squares or standard solve
                    return lib.linalg.solve(A, b)

            # Check for batching
            if matrix.ndim == 2:
                return solve_single(matrix, vector)
            else:
                # Batch loop
                out = []
                for A_i, b_i in zip(matrix, vector):
                    out.append(solve_single(A_i, b_i))
                return lib.array(out)
                
        else:
            raise ValueError(f"Unknown backend '{backend_name}'.")

    def solve_linear_system(self, matrix_data, vector_data, method='cholesky'):
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
        
        # Ensure vector has compatible data type (e.g. if matrix is float64, vector should be too)
        # This prevents backend errors
        
        if method == 'cholesky':
            result_data = self.cholesky_solve_helper(matrix_data, vector_data, backend_name, lib)
        else:
            # Standard direct solver
            if backend_name == 'torch':
                 b = vector_data
                 if b.ndim == matrix_data.ndim - 1:
                     b = b.unsqueeze(-1)
                 res = lib.linalg.solve(matrix_data, b)
                 result_data = res.squeeze(-1) if vector_data.ndim == matrix_data.ndim - 1 else res
            elif backend_name == 'numpy':
                # Handle batching for numpy if needed, though np.linalg.solve handles (B,N,N), (B,N)
                result_data = lib.linalg.solve(matrix_data, vector_data)
            else: # jax
                result_data = lib.linalg.solve(matrix_data, vector_data)

        return result_data

    def _get_node_update(self, node_tag, regularize=True, jitter=1e-6):
        b, H = self._compute_H_b(node_tag)
        
        # 1. Fuse indices to create matrix form
        variational_ind = b.inds
        # Maps for fusing indices
        map_H = {'rows': variational_ind, 'cols': [i + '_prime' for i in variational_ind]}
        map_b = {'cols': variational_ind}
        
        # Store sizes for unfusing later
        var_sizes = tuple(self.tn[node_tag].ind_size(i) for i in variational_ind)
        shape_map = {'cols': var_sizes} 

        # Fuse to dense structures
        H.fuse(map_H, inplace=True)
        b.fuse(map_b, inplace=True)
        
        matrix_data = H.to_dense(['rows'], ['cols'])
        vector = -b.to_dense(['cols'])

        # 2. Regularization (Efficient Diagonal Update)
        if regularize:
            backend, lib = self.get_backend(matrix_data)
            
            if backend == 'torch':
                # Efficient in-place diagonal update (view)
                # No extra memory allocation for Identity matrix
                matrix_data.diagonal().add_(jitter)
            
            elif backend == 'numpy':
                # Efficient numpy diagonal update
                rows, cols = lib.diag_indices_from(matrix_data)
                matrix_data[rows, cols] += jitter
                
            elif backend == 'jax':
                # JAX is immutable, so we use .at[].add()
                # indices for diagonal
                d_idx = lib.arange(matrix_data.shape[0])
                matrix_data = matrix_data.at[d_idx, d_idx].add(jitter)

        # 3. Solve Linear System
        tensor_node_data = self.solve_linear_system(matrix_data, vector)
        
        # 4. Wrap result back into Quimb Tensor and Unfuse
        # Use existing tags from the target node
        update_node = qt.Tensor(tensor_node_data, inds=['cols'], tags=self.tn[node_tag].tags)
        
        # Unfuse back to original indices using the shape map
        update_node.unfuse({'cols': variational_ind}, shape_map=shape_map, inplace=True)
        update_node.modify(tags = [node_tag])
        return update_node

    def update_node(self, tensor, node_tag):
        """
        Updates the tensor network by replacing the node with the given tag
        with the new tensor.
        """
        # Delete existing tensor(s) with this tag
        # Note: We assume node_tag uniquely identifies one tensor for replacement

        if node_tag in self.tn.tag_map:
            self.tn.delete(node_tag)
        
        # Add the new optimized tensor
        self.tn.add_tensor(tensor)

    def _get_trainable_nodes(self) -> List[str]:
        """
        Identifies all nodes (tags) that should be optimized.
        Excludes nodes with NOT_TRAINABLE_TAG.
        """
        trainable_tags = []
        for tensor in self.tn:
            # Check if this tensor is marked as not trainable
            if NOT_TRAINABLE_TAG in tensor.tags:
                continue
            
            # Find a unique tag for this tensor to use as identifier
            # We skip 'generic' tags if you have them, assuming standard unique tags like 'Node1', 'Node2' exist.
            # Here we just take the first tag that isn't the NT tag.
            valid_tags = list(tensor.tags)
            if valid_tags:
                # Prefer tags that look like "NodeX" or similar if possible, 
                # but taking the first valid one is usually safe in quimb if setup correctly.
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
            
            # Convert to torch
            y_pred_th = self._to_torch(y_pred)
            y_true_th = self._to_torch(y_true)

            # --- DEBUGGING PRINTS ---
            # print(f"DEBUG Eval | Pred: {y_pred_th.shape} ({y_pred_th.dtype}) | True: {y_true_th.shape} ({y_true_th.dtype})")
            
            # --- CRITICAL FIX: REMOVE view_as ---
            # Do NOT reshape y_pred here. 
            # If Pred is (100, 10) and True is (100, 1), they have different sizes (1000 vs 100).
            # Reshaping will crash. Let the metric function handle it.
            
            # Only reshape if element counts match (e.g. Regression: (100,1) vs (100,))
            if y_pred_th.numel() == y_true_th.numel():
                if y_pred_th.shape != y_true_th.shape:
                    y_pred_th = y_pred_th.view_as(y_true_th)

            # --- AGNOSTIC EXECUTION ---
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

    def evaluate(self, metrics: Dict[str, callable], verbose=False):
        """
        Iterates over the dataset and aggregates results from user metrics.
        
        Args:
            metrics: Dict mapping name -> callable(y_pred, y_true).
                     The callable should return a summable result (e.g. (loss_sum, count)).
        """
        # 1. Initialize Aggregates
        # We don't know the shape of the metric result yet, so we initialize on the first batch.
        aggregates = {}

        # 2. Iterate over Data
        # We iterate manually to handle the dictionary aggregation
        for i, batch_data in enumerate(self.data.data_mu_y):
            inputs, y_true = batch_data
            
            # Get dict of results for this batch
            batch_results = self._batch_evaluate(inputs, y_true, self.tn, metrics)
            
            # Aggregate
            if i == 0:
                aggregates = batch_results
            else:
                for name, res in batch_results.items():
                    # Polymorphic addition: works for floats, tensors, and tuples
                    # e.g. (loss, count) + (loss, count) = (total_loss, total_count)
                    if isinstance(res, tuple):
                        aggregates[name] = tuple(a + b for a, b in zip(aggregates[name], res))
                    else:
                        aggregates[name] += res

      
        final_scores = {}
        for name, val in aggregates.items():
            # If it's a tuple (value, count), auto-average it for display convenience
            if isinstance(val, (tuple, list)) and len(val) == 2:
                numerator, denominator = val
                if torch.is_tensor(denominator):
                    denominator = denominator.item()
                if denominator != 0:
                    final_scores[name] = (numerator / denominator).item() if torch.is_tensor(numerator) else (numerator / denominator)
                else:
                    final_scores[name] = 0.0
            else:
                # Just return the raw sum (e.g. total loss)
                final_scores[name] = val

        if verbose:
            print(f"Evaluation: {final_scores}")
            
        return final_scores

    def update_tn_node(self, node_tag, regularize, jitter):
        # 1. Calculate the Newton Step (Delta)
        # Solve (H) * delta = -Grad
        delta_tensor = self._get_node_update(
            node_tag, 
            regularize=regularize, 
            jitter=jitter
        )
        
        # 2. Get the current weight tensor
        current_tensor = self.tn[node_tag]
        
        # 3. Apply Update: w_new = w_old + delta
        # Quimb's (+) operator automatically aligns indices and adds the data
        new_tensor = current_tensor + delta_tensor
        
        # 4. Update the network with the new accumulated weights
        self.update_node(new_tensor, node_tag)

    def fit(self, n_epochs=1, regularize=True, jitter=1e-6, verbose=True, eval_metrics=None):
        """
        Main training loop (Alternating Least Squares / Newton Sweep).
        
        Args:
            eval_metrics: Dict of metric functions. Defaults to REGRESSION_METRICS if None.
        """
        # Default to Regression metrics if nothing provided
        if eval_metrics is None:
            eval_metrics = REGRESSION_METRICS

        trainable_nodes = self._get_trainable_nodes()
        
        # Standard DMRG-style sweep: Forward -> Backward (excluding ends to avoid double update)
        # This creates a smooth "wave" of updates: 1->2->3->2 (then 1 starts next epoch)
        back_sweep = trainable_nodes[-2:0:-1]
        full_sweep_order = trainable_nodes + back_sweep
        
        if verbose:
            print(f"Starting Fit: {n_epochs} epochs.")
            print(f"Sweep Order: {full_sweep_order}")

        # --- Initial Evaluation ---
        scores = self.evaluate(eval_metrics)
        if verbose:
            print(f"Init    | ", end="")
            print_metrics(scores)

        # --- Training Loop ---
        for epoch in range(n_epochs):
            
            # Optimization Sweep
            for node_tag in full_sweep_order:
                self.update_tn_node(node_tag, regularize, jitter)

            # Evaluation
            scores = self.evaluate(eval_metrics)
            
            if verbose:
                print(f"Epoch {epoch+1} | ", end="")
                print_metrics(scores)
                
        return scores
