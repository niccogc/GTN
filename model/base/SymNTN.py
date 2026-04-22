# type: ignore
"""
Symmetric NTN: Newton-based trainer for symmetric tensor networks.

Handles parameter tying where multiple nodes share the same tensor.
Key differences from NTN:
1. Gradient = dL * N_copies * dModel/dA
2. Hessian = d2L * sum_i(env_i ⊗ env_i) + dL * sum_{i≠j}(env_i ⊗ env_j)
"""
import re
import torch
import quimb.tensor as qt
from typing import List, Dict, Tuple, Optional
from model.base.NTN import NTN, NOT_TRAINABLE_TAG


class SymNTN(NTN):
    """
    NTN trainer for symmetric tensor networks.

    Requires model with:
        - sym_groups: Dict[str, List[int]] mapping canonical tag -> positions
        - position_to_canonical: Dict[int, str]
        - canonical_tensors: Dict[str, Tensor]
        - sync_tn_from_canonical(): rebuild TN after parameter update
    """

    def __init__(
        self,
        model,  # SymMPO2 or similar
        output_dims: List[str],
        input_dims: List[str],
        loss,
        data_stream,
        **kwargs,
    ):
        self.Left = model.Left
        self.Central = model.Central
        super().__init__(
            tn = (self.Left, self.Central),
            output_dims=output_dims,
            input_dims=input_dims,
            loss=loss,
            data_stream=data_stream,
            **kwargs,
        )

    def _batch_forward(self, inputs: List[qt.Tensor], tn, output_inds: List[str]) -> qt.Tensor:
        left_nodes, left_inputs = tn[0],inputs[:-1]
        central_node, central_inputs = tn[1] , inputs[-1]
        left_tn= left_nodes & left_inputs
        central_tn= central_node & central_inputs
        left_contraction = left_tn.contract()
        central_contraction = central_tn.contract()
        primed_left = left_contraction.copy()
        rank_ind = [i for i in left_contraction.inds if i != self.batch_dim]
        map = {i : i + "_prime" for i in rank_ind}
        primed_left = primed_left.reindex(map)
        y_tn = left_contraction & central_contraction & primed_left
        return y_tn.contract(output_inds=output_inds)

    # TODO: HERE IS WHERE I STOPPED
    def _batch_environment(self, inputs, tn: qt.TensorNetwork, target_tag: str, sum_over_batch: bool = False, sum_over_output: bool = False) -> qt.Tensor:
        return 
    
    def _batch_node_derivatives_symmetric(
        self, inputs, y_true, node_tag: str
    ) -> Tuple[qt.Tensor, qt.Tensor]:
        """
        Gradient and hessian for tied parameters.
        The nodes can look like this:
        A B C B A. symmetric MPS.

        Gradient =  (dL/dy * env_i) * N_symmetric (Example for A is 2, for B is 2, for C is 1) so for C actually we just use the standard environment calculation.
        Hessian = d2L/dy2 * env_i ⊗ env_i + dL/dy * env_ij. Where j is the position of the node symmetric to i.
        env_i means the network without node_i, env_ij is the network without node_ij
        """
        left = self.Left
        central = self.Central
        env_left = self._batch_environment(
            inputs, tn, target_tag=node_tag, sum_over_batch=False, sum_over_output=False
        )
        target_tensor = tn[node_tag]

        y_pred = self.forward_from_environment(
            env, node_tag=node_tag, node_tensor=target_tensor, sum_over_batch=False
        )

        dL_dy, d2L_dy2 = self.loss.get_derivatives(
            y_pred,
            y_true,
            backend=self.backend,
            batch_dim=self.batch_dim,
            output_dims=self.output_dimensions,
            return_hessian_diagonal=False,
            total_samples=self.train_data.samples,
        )

        grad_tn = env & dL_dy # This should be multiplied by the number of times node_tag is repeated (1 or 2.)
        node_inds = target_tensor.inds
        node_grad = grad_tn.contract(output_inds=node_inds)
        out_inds = self.output_dimensions

        out_row_inds = out_inds
        out_col_inds = [x + "_prime" for x in out_inds]

        d2L_tensor = qt.Tensor(
            d2L_dy2.data, inds=[self.batch_dim] + out_row_inds + out_col_inds
        )
        env_right = self._prime_indices_tensor(env, exclude_indices=[self.batch_dim])

        hess_tn = env & d2L_tensor & env_right

        node_inds = target_tensor.inds
        hess_out_inds = list(node_inds) + [f"{x}_prime" for x in node_inds]

        node_hess = hess_tn.contract(output_inds=hess_out_inds)
        num = re.sub(r"\D", "", node_tag)
        sym_pos = self.L -1 -num
        sym_tag = f"Node{sym_pos}"
        print(node_hess)
        # Clean up intermediate tensors to free memory
        del env, env_right, d2L_tensor, hess_tn, grad_tn, y_pred, dL_dy, d2L_dy2
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return node_grad, node_hess

    def _compute_H_b(self, node_tag: str):
        """
        Override to use symmetric derivatives for canonical parameters.
        """
        # Check if this is a canonical tag
        if node_tag in self.sym_model.sym_groups:
            J, H = self.compute_node_derivatives_symmetric(
                self.tn, node_tag, self.data.data_mu_y, sum_over_batches=True
            )
        else:
            J, H = super()._compute_H_b(node_tag)
        return J, H

    def compute_node_derivatives_symmetric(
        self,
        tn: qt.TensorNetwork,
        node_tag: str,
        input_generator,
        sum_over_batches=True,
    ):
        """Compute gradient/hessian for symmetric (tied) parameters."""
        if sum_over_batches:
            return self._sum_over_batches(
                self._batch_node_derivatives_symmetric,
                input_generator,
                node_tag=node_tag,
            )
        else:
            return self._concat_tuple_over_batches(
                self._batch_node_derivatives_symmetric,
                input_generator,
                node_tag=node_tag,
            )

    def update_node(self, tensor: qt.Tensor, node_tag: str):
        """
        Override to update canonical tensor and sync all copies.
        """
        if node_tag in self.sym_model.sym_groups:
            # Update canonical tensor
            self.sym_model.canonical_tensors[node_tag] = tensor.data.clone()
            # Rebuild TN with all copies
            self.sym_model.sync_tn_from_canonical()
            self.tn = self.sym_model.tn
        else:
            super().update_node(tensor, node_tag)
