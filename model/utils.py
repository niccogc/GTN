import torch

def metric_mse(y_pred, y_true):
    """Mean Squared Error."""
    # Robust Reshape: Ensure target matches pred
    if y_pred.shape != y_true.shape:
        y_true = y_true.view_as(y_pred)
        
    return torch.sum((y_true - y_pred) ** 2), y_true.numel()
import torch

def metric_accuracy(y_pred, y_true):
    """
    y_pred: (Batch, Classes) -> Logits
    y_true: (Batch, Classes) -> One-Hot OR (Batch,) -> Indices
    """
    # Get Class Indices from Logits
    preds = y_pred.argmax(dim=1)
    
    # Handle One-Hot Ground Truth
    if y_true.ndim > 1 and y_true.shape[1] > 1:
        truth = y_true.argmax(dim=1)
    else:
        truth = y_true.view(-1)
        
    correct = (preds == truth).sum()
    return correct, truth.shape[0]

def metric_cross_entropy(y_pred, y_true):
    """
    y_pred: (Batch, Classes)
    y_true: One-Hot Float (Batch, Classes) OR Long Indices (Batch,)
    """
    # PyTorch CrossEntropy supports both:
    # 1. Target = Long Indices (N,)
    # 2. Target = Float Probabilities (N, C) <--- We use this for One-Hot
    
    if y_true.ndim == 1 or y_true.shape[1] == 1:
        # Indices case
        target = y_true.view(-1).long()
    else:
        # One-Hot case (must be float)
        target = y_true.float()
        
    loss = torch.nn.functional.cross_entropy(y_pred, target, reduction='sum')
    return loss, y_true.shape[0]

# Pre-packaged dictionary
CLASSIFICATION_METRICS = {
    'accuracy': metric_accuracy,
    'loss': metric_cross_entropy
}

def metric_r2_components(y_pred, y_true):
    """R2 Components."""
    # Robust Reshape
    if y_pred.shape != y_true.shape:
        y_true = y_true.view_as(y_pred)
        
    s_y = torch.sum(y_true)
    s_yy = torch.sum(y_true ** 2)
    s_res = torch.sum((y_true - y_pred) ** 2)
    n = torch.tensor(y_true.numel(), device=y_true.device)
    
    return s_y, s_yy, s_res, n
# --- 2. Pre-Packaged Dictionaries ---

REGRESSION_METRICS = {
    'mse': metric_mse,
    'r2_stats': metric_r2_components
}
# --- 3. Post-Processing Helpers ---

def compute_final_r2(metrics_result):
    """
    Takes the aggregated 'r2_stats' result (tuple of 4) and computes the scalar R^2.
    """
    if 'r2_stats' not in metrics_result:
        return None

    # Unpack the tuple: (Sum_Y, Sum_Y^2, Sum_Res^2, N)
    s_y, s_yy, s_res, n = metrics_result['r2_stats']
    
    # Ensure scalars
    n = n.item() if torch.is_tensor(n) else n
    if n == 0: return 0.0
    
    # Global Mean
    y_mean = s_y / n
    
    # Total Sum of Squares = Sum(y^2) - n * mean^2
    ss_tot = s_yy - (n * (y_mean ** 2))
    
    # R^2 = 1 - (SS_res / SS_tot)
    if ss_tot == 0:
        return 0.0
        
    r2 = 1 - (s_res / ss_tot)
    
    # Clean up tensor if needed
    return r2.item() if torch.is_tensor(r2) else r2

def print_metrics(metrics_result):
    """
    Pretty prints metrics, handling special post-processing for R2.
    """
    clean_metrics = {}
    
    for k, v in metrics_result.items():
        if k == 'r2_stats':
            clean_metrics['R2'] = compute_final_r2(metrics_result)
        elif isinstance(v, (tuple, list)) and len(v) == 2:
            # Standard (numerator, denominator)
            num, den = v
            val = (num / den).item() if torch.is_tensor(num) else (num / den)
            clean_metrics[k] = val
        else:
            # Raw value
            clean_metrics[k] = v.item() if torch.is_tensor(v) else v
            
    # Format string
    out_str = " | ".join([f"{k}: {v:.5f}" for k, v in clean_metrics.items()])
    print(out_str)
    return clean_metrics
