# type: ignore
"""
Simple CMPO2_NTN without any caching - just uses native NTN functions.
"""
from model.NTN import NTN

class SimpleCMPO2_NTN(NTN):
    """
    CMPO2 implementation without environment caching.
    Uses only the native NTN functions for everything.
    
    For CMPO2, we have two MPS layers (pixels and patches) with tags like:
    - 0_Pi, 1_Pi, 2_Pi (pixel MPS)
    - 0_Pa, 1_Pa, 2_Pa (patch MPS)
    
    We need to train each tensor individually, not by site tag.
    """
    def __init__(self, *args, psi=None, phi=None, **kwargs):
        super().__init__(*args, **kwargs)
        # Store references to the original MPS objects
        self.psi = psi
        self.phi = phi
    
    def fit(self, n_epochs=1, regularize=True, jitter=1e-6, verbose=True, eval_metrics=None):
        """
        Override fit to normalize the tensor network after each sweep.
        This prevents the explosion of node norms during training.
        """
        import torch
        from model.utils import REGRESSION_METRICS
        
        # Default to Regression metrics if nothing provided
        if eval_metrics is None:
            eval_metrics = REGRESSION_METRICS

        if not isinstance(jitter, list):
            jitter = [jitter]*n_epochs
        trainable_nodes = self._get_trainable_nodes()
        
        # Standard DMRG-style sweep: Forward -> Backward
        back_sweep = trainable_nodes[-2:0:-1]
        full_sweep_order = trainable_nodes + back_sweep
        
        def print_metrics(scores):
            for k, v in scores.items():
                print(f"{k}: {v:.5f} | ", end="")
            print()
        
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
                self.update_tn_node(node_tag, regularize, jitter[epoch])
                
                # Normalize after EACH node update to prevent explosion
                # Get all _Pi tensors and normalize them as a group
                pi_tensors = [self.tn[tag] for tag in self._get_trainable_nodes() if '_Pi' in tag]
                if pi_tensors:
                    total_norm_sq = sum(torch.sum(t.data ** 2).item() for t in pi_tensors)
                    norm = total_norm_sq ** 0.5
                    if norm > 0:
                        for t in pi_tensors:
                            t.modify(data=t.data / norm)
                
                # Get all _Pa tensors and normalize them as a group
                pa_tensors = [self.tn[tag] for tag in self._get_trainable_nodes() if '_Pa' in tag]
                if pa_tensors:
                    total_norm_sq = sum(torch.sum(t.data ** 2).item() for t in pa_tensors)
                    norm = total_norm_sq ** 0.5
                    if norm > 0:
                        for t in pa_tensors:
                            t.modify(data=t.data / norm)

            # Evaluation
            scores = self.evaluate(eval_metrics)
            
            if verbose:
                print(f"Epoch {epoch+1} | ", end="")
                print_metrics(scores)
                
        return scores
    
    def _get_trainable_nodes(self):
        """
        Override to return individual tensor tags (0_Pi, 0_Pa, etc.)
        instead of site tags (I0, I1, etc.).
        
        For CMPO2, each site has TWO tensors, so we need to update them separately.
        """
        # Get all tensors that are NOT input tensors
        trainable_tags = []
        for tensor in self.tn.tensors:
            # Skip input tensors (they have 'input_' in their tags)
            if any('input_' in str(tag) for tag in tensor.tags):
                continue
            
            # Find the specific tensor tag (like '0_Pi' or '1_Pa')
            for tag in tensor.tags:
                tag_str = str(tag)
                # Look for tags that end with _Pi or _Pa
                if '_Pi' in tag_str or '_Pa' in tag_str:
                    if tag_str not in trainable_tags:
                        trainable_tags.append(tag_str)
                    break
        
        # Sort them in a reasonable order: 0_Pi, 0_Pa, 1_Pi, 1_Pa, ...
        def sort_key(tag):
            if '_Pi' in tag:
                num = int(tag.split('_')[0])
                return (num, 0)  # Pi comes first
            else:  # _Pa
                num = int(tag.split('_')[0])
                return (num, 1)  # Pa comes second
        
        trainable_tags.sort(key=sort_key)
        return trainable_tags
