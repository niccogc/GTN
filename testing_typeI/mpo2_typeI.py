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
    """Custom NTN that computes derivatives w.r.t. ensemble predictions."""

    def __init__(self, ensemble_forward_fn: Callable, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ensemble_forward_fn = ensemble_forward_fn

    def _compute_y_pred_for_node_derivative(self, env, target_tensor, inputs, y_true):
        """Use ensemble prediction instead of individual model prediction."""
        return self.ensemble_forward_fn(inputs, y_true)


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
    """Coordinator for multiple NTN instances."""

    def __init__(
        self,
        max_sites: int,
        bond_dim: int,
        phys_dim: int,
        output_dim: int,
        loss,
        X_data: torch.Tensor,
        y_data: torch.Tensor,
        batch_size: int = 32,
        output_site: Optional[int] = None,
        init_strength: float = 0.001,
        method: str = "cholesky",
        not_trainable_nodes: List[str] = [],
        lambda_reg: float = 1e-3,
    ):
        self.max_sites = max_sites
        self.ntns = []
        self.input_loaders = []

        for L in range(1, max_sites + 1):
            tn, input_labels, input_dims, output_dims = create_simple_mps(
                L=L,
                bond_dim=bond_dim,
                phys_dim=phys_dim,
                output_dim=output_dim,
                output_site=output_site,
                init_strength=init_strength,
            )

            loader = Inputs(
                inputs=[X_data],
                outputs=[y_data],
                outputs_labels=output_dims,
                input_labels=input_labels,
                batch_dim="s",
                batch_size=batch_size,
            )

            ntn = NTN(
                tn=tn,
                output_dims=output_dims,
                input_dims=input_dims,
                loss=loss,
                data_stream=loader,
                method=method,
                not_trainable_nodes=not_trainable_nodes,
                lambda_reg=lambda_reg,
            )

            self.input_loaders.append(loader)
            self.ntns.append(ntn)

        self.loss = loss
        self.output_dims = output_dims
        self.batch_dim = "s"
        self.lambda_reg = lambda_reg

    def _get_all_trainable_nodes(self) -> List[tuple]:
        """Get all trainable nodes as (ntn_idx, node_tag) tuples."""
        all_nodes = []
        for ntn_idx, ntn in enumerate(self.ntns):
            trainable = ntn._get_trainable_nodes()
            for tag in trainable:
                all_nodes.append((ntn_idx, tag))
        return all_nodes

    def forward(self, data_stream: Inputs) -> torch.Tensor:
        """Ensemble forward: sum outputs from all NTN instances."""
        total_output = None

        for ntn, loader in zip(self.ntns, self.input_loaders):
            # Rewind both loaders to start
            loader.rewind()
            data_stream.rewind()

            # Accumulate outputs batch by batch
            batch_outputs = []
            for batch_idx in range(len(data_stream)):
                inputs_dict = data_stream[batch_idx]
                loader_inputs = loader[batch_idx]

                # Use loader's inputs with proper labels for this NTN
                ntn_output = ntn.forward(loader_inputs)
                batch_outputs.append(ntn_output)

            # Concatenate batches
            ntn_full_output = torch.cat(batch_outputs, dim=0)

            if total_output is None:
                total_output = ntn_full_output
            else:
                total_output = total_output + ntn_full_output

        data_stream.rewind()
        return total_output

    def evaluate(self, metrics, data_stream: Inputs):
        """Evaluate ensemble performance."""
        y_pred = self.forward(data_stream)

        # Get true labels
        all_y_true = []
        data_stream.rewind()
        for batch_idx in range(len(data_stream)):
            batch = data_stream[batch_idx]
            y_true = batch[self.output_dims[0]]
            all_y_true.append(y_true)
        y_true = torch.cat(all_y_true, dim=0)
        data_stream.rewind()

        # Compute loss
        loss_value = self.loss(y_pred, y_true)

        # Compute metrics
        scores = {"loss": loss_value.item()}
        for metric_name, metric_fn in metrics.items():
            scores[metric_name] = metric_fn(y_pred, y_true).item()

        return scores

    def fit(
        self,
        n_epochs: int = 1,
        regularize: bool = True,
        jitter: float = 1e-6,
        verbose: bool = True,
        eval_metrics=None,
        val_data=None,
        test_data=None,
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

        if verbose:
            print(f"Starting Fit: {n_epochs} epochs")
            print(
                f"Models: {[(i + 1, len(ntn._get_trainable_nodes())) for i, ntn in enumerate(self.ntns)]}"
            )
            print(f"Total nodes: {len(all_nodes)}")

        best_val_quality = float("-inf")
        patience_counter = 0

        for epoch in range(n_epochs):
            for ntn_idx, node_tag in all_nodes:
                ntn = self.ntns[ntn_idx]
                ntn.update_tn_node(node_tag, regularize, jitter_schedule[epoch], adaptive_jitter)

            scores_train = self.ntns[0].evaluate(eval_metrics, data_stream=self.input_loaders[0])

            if val_data:
                scores_val = self.ntns[0].evaluate(eval_metrics, data_stream=val_data)
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
                if val_data:
                    print(f"        | Val:   ", end="")
                    print_metrics(scores_val)
                    print(f"{marker}", end="")
                print()

            if patience and patience_counter >= patience:
                if verbose:
                    print(f"Early stopping at epoch {epoch + 1}")
                break

        return scores_train, scores_val if val_data else scores_train
