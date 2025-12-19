# type: ignore
"""
Tensor Network Loss Functions

This module provides loss functions compatible with tensor network optimization.
Each loss inherits from both a base TNLoss class and the corresponding PyTorch loss,
providing a custom get_derivatives method that returns gradients and Hessians
in the correct quimb tensor format.
"""

import torch
import torch.nn as nn
import quimb.tensor as qt
from typing import Tuple, Optional, List


class TNLoss:
    """
    Base class for losses compatible with Tensor Network optimization.
    
    All TN losses must implement get_derivatives() which computes both
    gradient and Hessian w.r.t. the predictions.
    
    Attributes:
        use_diagonal_hessian: If True, use diagonal Hessian approximation.
                             If False, compute full Hessian matrix.
                             Subclasses can override based on loss characteristics.
    """
    
    use_diagonal_hessian = True  # Default: use diagonal approximation
    
    def get_derivatives(
        self, 
        y_pred: qt.Tensor, 
        y_true: qt.Tensor,
        backend: str = 'numpy',
        batch_dim: str = 'batch',
        output_dims: Optional[List[str]] = None,
        return_hessian_diagonal: Optional[bool] = None
    ) -> Tuple[qt.Tensor, qt.Tensor]:
        """
        Compute gradient and Hessian of loss w.r.t predictions.
        
        Args:
            y_pred: quimb.Tensor with indices [batch_dim, *output_dims]
                   Shape: (sample, out_shape...)
            y_true: quimb.Tensor with indices [batch_dim, *output_dims]
                   Shape: (sample, out_shape...)
            backend: Backend to use for returned tensors ('numpy', 'torch', 'jax')
            batch_dim: Name of the batch dimension index
            output_dims: List of output dimension index names
            return_hessian_diagonal: If True, return diagonal Hessian approximation
            
        Returns:
            grad: quimb.Tensor with shape (sample, out_shape...)
                 Indices: [batch_dim, *output_dims]
            hess: quimb.Tensor with Hessian
                 If diagonal: shape (sample, out_shape...), indices: [batch_dim, *output_dims]
                 If full: shape (sample, out_shape..., out_shape..._prime)
                         indices: [batch_dim, *output_dims, *output_dims_prime]
        """
        # Use instance property if not explicitly specified
        if return_hessian_diagonal is None:
            return_hessian_diagonal = self.use_diagonal_hessian
        
        raise NotImplementedError("Subclasses must implement get_derivatives()")
    def _to_torch(self, tensor, requires_grad=False):
        """Convert quimb tensor or array to torch tensor."""
        data = tensor.data if isinstance(tensor, qt.Tensor) else tensor
        if not torch.is_tensor(data):
            data = torch.from_numpy(data)
        if not data.is_floating_point():
            data = data.float()
        if requires_grad:
            data.requires_grad_(True)
        return data


class MSELoss(nn.MSELoss, TNLoss):
    """
    Mean Squared Error Loss for Tensor Networks.
    
    Inherits from both torch.nn.MSELoss and TNLoss to provide
    standard PyTorch loss computation and custom derivative methods.
    
    For MSE: L = (1/N) * sum((y_pred - y_true)^2)
    
    Gradient: dL/dy_pred = 2 * (y_pred - y_true) / N
    Hessian (diagonal): d²L/dy_pred² = 2 / N (constant)
    
    Note: MSE has a diagonal Hessian by default (outputs are independent).
    """
    
    use_diagonal_hessian = True  # MSE outputs are independent
    
    def __init__(self, reduction='mean', use_diagonal_hessian=True):
        """
        Initialize MSE Loss.
        
        Args:
            reduction: 'mean', 'sum', or 'none' (follows PyTorch convention)
            use_diagonal_hessian: Whether to use diagonal approximation (default True)
        """
        nn.MSELoss.__init__(self, reduction=reduction)
        TNLoss.__init__(self)
        self.use_diagonal_hessian = use_diagonal_hessian
    
    def get_derivatives(
        self, 
        y_pred: qt.Tensor, 
        y_true: qt.Tensor,
        backend: str = 'numpy',
        batch_dim: str = 'batch',
        output_dims: Optional[List[str]] = None,
        return_hessian_diagonal: Optional[bool] = None
    ) -> Tuple[qt.Tensor, qt.Tensor]:
        """
        Compute MSE gradient and Hessian using PyTorch autograd.
        
        Returns gradients and Hessians as quimb tensors with proper indices.
        """
        # Use instance property if not explicitly specified
        if return_hessian_diagonal is None:
            return_hessian_diagonal = self.use_diagonal_hessian
        
        # Use instance property if not explicitly specified
        if return_hessian_diagonal is None:
            return_hessian_diagonal = self.use_diagonal_hessian
        
        
        # 1. Prepare Data
        y_pred_th = self._to_torch(y_pred, requires_grad=True)
        y_true_th = self._to_torch(y_true, requires_grad=False)
        if y_pred_th.device != y_true_th.device:
            y_true_th = y_true_th.to(y_pred_th.device)

        batch_sz = y_pred_th.shape[0]
        y_flat = y_pred_th.view(batch_sz, -1)
        num_outputs = y_flat.shape[1]

        # 2. Compute Loss
        loss_val = nn.MSELoss.__call__(self, y_pred_th, y_true_th)
        
        # 3. First Derivative (Gradient)
        grad_th = torch.autograd.grad(loss_val, y_pred_th, create_graph=True)[0]
        grad_flat = grad_th.view(batch_sz, -1)
        
        # 4. Second Derivative (Hessian)
        if return_hessian_diagonal:
            # Diagonal Hessian: compute d²L/dy_i² for each output dimension
            hess_cols = []
            for i in range(num_outputs):
                grad_sum = grad_flat[:, i].sum()
                h_i = torch.autograd.grad(grad_sum, y_pred_th, retain_graph=True)[0]
                h_i_flat = h_i.view(batch_sz, -1)
                hess_cols.append(h_i_flat[:, i])
            
            hess_th = torch.stack(hess_cols, dim=1).view(y_pred_th.shape)
            
            # Indices match y_pred for diagonal
            hess_inds = y_pred.inds if isinstance(y_pred, qt.Tensor) else None

        else:
            # Full Per-Sample Hessian: (Batch, Out, Out)
            hess_rows = []
            for i in range(num_outputs):
                grad_sum = grad_flat[:, i].sum()
                h_row = torch.autograd.grad(grad_sum, y_pred_th, retain_graph=True)[0]
                hess_rows.append(h_row.view(batch_sz, -1))
            
            # Shape: (Batch, Out, Out)
            hess_th = torch.stack(hess_rows, dim=0).permute(1, 0, 2)
            
            # Create proper indices for the full matrix [batch, out, out_prime]
            if isinstance(y_pred, qt.Tensor):
                out_inds = [i for i in y_pred.inds if i != batch_dim]
                out_inds_prime = [i + "_prime" for i in out_inds]
                hess_inds = [batch_dim] + out_inds + out_inds_prime
            else:
                hess_inds = None

        # 5. Convert to target backend
        if backend == 'numpy':
            grad_data = grad_th.detach().numpy()
            hess_data = hess_th.detach().numpy()
        else:  # torch or jax
            grad_data = grad_th
            hess_data = hess_th
        
        # 6. Wrap in quimb tensors with proper indices
        grad_inds = y_pred.inds if isinstance(y_pred, qt.Tensor) else None
        return qt.Tensor(grad_data, inds=grad_inds), qt.Tensor(hess_data, inds=hess_inds)  # type: ignore


class MAELoss(nn.L1Loss, TNLoss):
    """
    Mean Absolute Error (L1) Loss for Tensor Networks.
    
    For MAE: L = (1/N) * sum(|y_pred - y_true|)
    
    Gradient: dL/dy_pred = sign(y_pred - y_true) / N
    Hessian: d²L/dy_pred² ≈ 0 (undefined at y_pred = y_true, approximated as small constant)
    
    Note: MAE has a diagonal Hessian (outputs are independent).
    """
    
    use_diagonal_hessian = True  # MAE outputs are independent
    
    def __init__(self, reduction='mean', hessian_eps=1e-6, use_diagonal_hessian=True):
        """
        Initialize MAE Loss.
        
        Args:
            reduction: 'mean', 'sum', or 'none'
            hessian_eps: Small constant for Hessian approximation (L1 has zero second derivative)
            use_diagonal_hessian: Whether to use diagonal approximation (default True)
        """
        nn.L1Loss.__init__(self, reduction=reduction)
        TNLoss.__init__(self)
        self.hessian_eps = hessian_eps
        self.use_diagonal_hessian = use_diagonal_hessian
    
    def get_derivatives(
        self, 
        y_pred: qt.Tensor, 
        y_true: qt.Tensor,
        backend: str = 'numpy',
        batch_dim: str = 'batch',
        output_dims: Optional[List[str]] = None,
        return_hessian_diagonal: Optional[bool] = None
    ) -> Tuple[qt.Tensor, qt.Tensor]:
        """
        Compute MAE gradient and approximate Hessian.
        
        Note: L1 loss has zero second derivative almost everywhere,
        so we return a small constant for numerical stability.
        """
        # Use instance property if not explicitly specified
        if return_hessian_diagonal is None:
            return_hessian_diagonal = self.use_diagonal_hessian
        
        # 1. Prepare Data
        y_pred_th = self._to_torch(y_pred, requires_grad=True)
        y_true_th = self._to_torch(y_true, requires_grad=False)
        if y_pred_th.device != y_true_th.device:
            y_true_th = y_true_th.to(y_pred_th.device)

        batch_sz = y_pred_th.shape[0]

        # 2. Compute Loss
        loss_val = nn.L1Loss.__call__(self, y_pred_th, y_true_th)
        
        # 3. First Derivative (Gradient)
        grad_th = torch.autograd.grad(loss_val, y_pred_th, create_graph=False)[0]
        
        # 4. Second Derivative (Hessian)
        # L1 has zero second derivative, use small constant for stability
        hess_th = torch.full_like(y_pred_th, self.hessian_eps)
        
        if not return_hessian_diagonal:
            # Full Hessian is diagonal with small constants
            # Create (Batch, Out, Out) with diagonal elements
            if isinstance(y_pred, qt.Tensor):
                out_inds = [i for i in y_pred.inds if i != batch_dim]
                out_inds_prime = [i + "_prime" for i in out_inds]
                
                # Create diagonal matrix
                batch_size = y_pred_th.shape[0]
                out_size = y_pred_th.shape[1] if y_pred_th.ndim > 1 else 1
                hess_full = torch.zeros(batch_size, out_size, out_size, 
                                       dtype=y_pred_th.dtype, device=y_pred_th.device)
                for i in range(out_size):
                    hess_full[:, i, i] = self.hessian_eps
                hess_th = hess_full
                hess_inds = [batch_dim] + out_inds + out_inds_prime
            else:
                hess_inds = None
        else:
            hess_inds = y_pred.inds if isinstance(y_pred, qt.Tensor) else None

        # 5. Convert to target backend
        if backend == 'numpy':
            grad_data = grad_th.detach().numpy()
            hess_data = hess_th.detach().numpy() if torch.is_tensor(hess_th) else hess_th
        else:  # torch
            grad_data = grad_th
            hess_data = hess_th
        
        # 6. Wrap in quimb tensors
        grad_inds = y_pred.inds if isinstance(y_pred, qt.Tensor) else None
        return qt.Tensor(grad_data, inds=grad_inds), qt.Tensor(hess_data, inds=hess_inds)  # type: ignore


class HuberLoss(nn.HuberLoss, TNLoss):
    """
    Huber Loss for Tensor Networks (robust regression).
    
    Combines MSE (for small errors) and MAE (for large errors).
    
    For |error| <= delta: L = 0.5 * error²
    For |error| > delta:  L = delta * (|error| - 0.5 * delta)
    
    This provides robustness to outliers while being smooth near zero.
    
    Note: Huber has a diagonal Hessian (outputs are independent).
    """
    
    use_diagonal_hessian = True  # Huber outputs are independent
    
    def __init__(self, reduction='mean', delta=1.0, use_diagonal_hessian=True):
        """
        Initialize Huber Loss.
        
        Args:
            reduction: 'mean', 'sum', or 'none'
            delta: Threshold for switching between quadratic and linear
            use_diagonal_hessian: Whether to use diagonal approximation (default True)
        """
        nn.HuberLoss.__init__(self, reduction=reduction, delta=delta)
        TNLoss.__init__(self)
        self.use_diagonal_hessian = use_diagonal_hessian
    
    def get_derivatives(
        self, 
        y_pred: qt.Tensor, 
        y_true: qt.Tensor,
        backend: str = 'numpy',
        batch_dim: str = 'batch',
        output_dims: Optional[List[str]] = None,
        return_hessian_diagonal: Optional[bool] = None
    ) -> Tuple[qt.Tensor, qt.Tensor]:
        """
        Compute Huber loss gradient and Hessian using PyTorch autograd.
        """
        # Use instance property if not explicitly specified
        if return_hessian_diagonal is None:
            return_hessian_diagonal = self.use_diagonal_hessian
        
        # 1. Prepare Data
        y_pred_th = self._to_torch(y_pred, requires_grad=True)
        y_true_th = self._to_torch(y_true, requires_grad=False)
        if y_pred_th.device != y_true_th.device:
            y_true_th = y_true_th.to(y_pred_th.device)

        batch_sz = y_pred_th.shape[0]
        y_flat = y_pred_th.view(batch_sz, -1)
        num_outputs = y_flat.shape[1]

        # 2. Compute Loss
        loss_val = nn.HuberLoss.__call__(self, y_pred_th, y_true_th)
        
        # 3. First Derivative (Gradient)
        grad_th = torch.autograd.grad(loss_val, y_pred_th, create_graph=True)[0]
        grad_flat = grad_th.view(batch_sz, -1)
        
        # 4. Second Derivative (Hessian)
        if return_hessian_diagonal:
            hess_cols = []
            for i in range(num_outputs):
                grad_sum = grad_flat[:, i].sum()
                h_i = torch.autograd.grad(grad_sum, y_pred_th, retain_graph=True)[0]
                h_i_flat = h_i.view(batch_sz, -1)
                hess_cols.append(h_i_flat[:, i])
            
            hess_th = torch.stack(hess_cols, dim=1).view(y_pred_th.shape)
            hess_inds = y_pred.inds if isinstance(y_pred, qt.Tensor) else None

        else:
            # Full Per-Sample Hessian
            hess_rows = []
            for i in range(num_outputs):
                grad_sum = grad_flat[:, i].sum()
                h_row = torch.autograd.grad(grad_sum, y_pred_th, retain_graph=True)[0]
                hess_rows.append(h_row.view(batch_sz, -1))
            
            hess_th = torch.stack(hess_rows, dim=0).permute(1, 0, 2)
            
            if isinstance(y_pred, qt.Tensor):
                out_inds = [i for i in y_pred.inds if i != batch_dim]
                out_inds_prime = [i + "_prime" for i in out_inds]
                hess_inds = [batch_dim] + out_inds + out_inds_prime
            else:
                hess_inds = None

        # 5. Convert to target backend
        if backend == 'numpy':
            grad_data = grad_th.detach().numpy()
            hess_data = hess_th.detach().numpy()
        else:
            grad_data = grad_th
            hess_data = hess_th
        
        # 6. Wrap in quimb tensors
        grad_inds = y_pred.inds if isinstance(y_pred, qt.Tensor) else None
        return qt.Tensor(grad_data, inds=grad_inds), qt.Tensor(hess_data, inds=hess_inds)  # type: ignore


class CrossEntropyLoss(nn.CrossEntropyLoss, TNLoss):
    """
    Cross Entropy Loss for Tensor Networks (classification).
    
    For multi-class classification with C classes:
    - Input (logits): shape (batch, C)
    - Target: class indices shape (batch,) OR one-hot shape (batch, C)
    
    Loss: L = -sum(y_true * log(softmax(y_pred)))
    
    Gradient: dL/dz_i = p_i - y_i  where p = softmax(logits), y = one-hot target
    Hessian: d²L/dz_i dz_j = p_i * (δ_ij - p_j)  (full matrix, not diagonal!)
    
    Note: The Hessian is inherently a FULL MATRIX for cross-entropy because
    changing one logit affects all class probabilities through the softmax.
    Using diagonal approximation is possible but not recommended.
    """
    
    use_diagonal_hessian = False  # CrossEntropy needs full Hessian (softmax coupling)
    
    def __init__(self, reduction='mean', ignore_index=-100, label_smoothing=0.0, use_diagonal_hessian=False):
        """
        Initialize Cross Entropy Loss.
        
        Args:
            reduction: 'mean', 'sum', or 'none'
            ignore_index: Target value to ignore (default -100)
            label_smoothing: Label smoothing factor [0, 1]
            use_diagonal_hessian: Whether to use diagonal approximation (default False, not recommended)
        """
        nn.CrossEntropyLoss.__init__(
            self, 
            reduction=reduction,
            ignore_index=ignore_index,
            label_smoothing=label_smoothing
        )
        TNLoss.__init__(self)
        self.use_diagonal_hessian = use_diagonal_hessian
    
    def get_derivatives(
        self, 
        y_pred: qt.Tensor, 
        y_true: qt.Tensor,
        backend: str = 'numpy',
        batch_dim: str = 'batch',
        output_dims: Optional[List[str]] = None,
        return_hessian_diagonal: Optional[bool] = None
    ) -> Tuple[qt.Tensor, qt.Tensor]:
        """
        Compute Cross Entropy gradient and Hessian using PyTorch autograd.
        
        Note: For cross-entropy, the Hessian is inherently a full matrix due to
        softmax coupling. If return_hessian_diagonal=True, we return a diagonal
        approximation (just the diagonal elements), but the full matrix is recommended.
        
        Args:
            y_pred: Logits, shape (batch, num_classes)
                   Indices: [batch_dim, class_dim]
            y_true: Either class indices (batch,) or one-hot (batch, num_classes)
                   For class indices: Indices: [batch_dim]
                   For one-hot: Indices: [batch_dim, class_dim]
        """
        # Use instance property if not explicitly specified
        if return_hessian_diagonal is None:
            return_hessian_diagonal = self.use_diagonal_hessian
        
        # 1. Prepare Data
        y_pred_th = self._to_torch(y_pred, requires_grad=True)
        y_true_th = self._to_torch(y_true, requires_grad=False)
        if y_pred_th.device != y_true_th.device:
            y_true_th = y_true_th.to(y_pred_th.device)

        batch_sz = y_pred_th.shape[0]
        num_classes = y_pred_th.shape[1]
        
        # Determine if y_true is class indices or one-hot
        is_class_indices = (y_true_th.ndim == 1) or (y_true_th.shape[1] == 1)
        
        if is_class_indices:
            # Convert to long indices if needed
            if y_true_th.ndim == 2 and y_true_th.shape[1] == 1:
                y_true_indices = y_true_th.squeeze(1).long()
            else:
                y_true_indices = y_true_th.long()
        else:
            # One-hot: convert to indices
            y_true_indices = y_true_th.argmax(dim=1)

        # 2. Compute Loss
        loss_val = nn.CrossEntropyLoss.__call__(self, y_pred_th, y_true_indices)
        
        # 3. First Derivative (Gradient) using autograd
        grad_th = torch.autograd.grad(loss_val, y_pred_th, create_graph=True)[0]
        
        # 4. Second Derivative (Hessian) using autograd
        if return_hessian_diagonal:
            # Diagonal approximation: compute only diagonal elements
            hess_cols = []
            for i in range(num_classes):
                grad_sum = grad_th[:, i].sum()
                h_i = torch.autograd.grad(grad_sum, y_pred_th, retain_graph=True)[0]
                hess_cols.append(h_i[:, i])
            
            hess_th = torch.stack(hess_cols, dim=1)
            hess_inds = y_pred.inds if isinstance(y_pred, qt.Tensor) else None

        else:
            # Full Per-Sample Hessian: (Batch, Classes, Classes)
            hess_rows = []
            for i in range(num_classes):
                grad_sum = grad_th[:, i].sum()
                h_row = torch.autograd.grad(grad_sum, y_pred_th, retain_graph=True)[0]
                hess_rows.append(h_row)
            
            # Shape: (Batch, Classes, Classes)
            hess_th = torch.stack(hess_rows, dim=1)
            
            # Create proper indices for the full matrix [batch, class, class_prime]
            if isinstance(y_pred, qt.Tensor):
                # Get the class dimension (the one that's not batch_dim)
                class_dim = [i for i in y_pred.inds if i != batch_dim][0]
                hess_inds = [batch_dim, class_dim, class_dim + "_prime"]
            else:
                hess_inds = None

        # 5. Convert to target backend
        if backend == 'numpy':
            grad_data = grad_th.detach().numpy()
            hess_data = hess_th.detach().numpy()
        else:  # torch
            grad_data = grad_th
            hess_data = hess_th
        
        
        # 6. Wrap in quimb tensors
        grad_inds = y_pred.inds if isinstance(y_pred, qt.Tensor) else None
        return qt.Tensor(grad_data, inds=grad_inds), qt.Tensor(hess_data, inds=hess_inds)  # type: ignore
