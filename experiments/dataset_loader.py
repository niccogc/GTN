# type: ignore
"""
Dataset loader utility for production training scripts.
Uses UCI ML Repository for loading datasets.
"""
import torch
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.load_ucirepo import get_ucidata, datasets as uci_datasets


def load_dataset(dataset_name, n_samples=None, seed=0, val_split=0.2, test_split=0.2, device='cpu', cap=50):
    """
    Load a dataset by name from UCI ML Repository.
    
    Args:
        dataset_name: Name of dataset to load (e.g., 'abalone', 'iris')
        n_samples: Number of samples to use (None = use all) - NOT IMPLEMENTED YET
        seed: Random seed - IGNORED (splits are handled by get_ucidata with fixed seed=42)
        val_split: Fraction for validation - IGNORED (splits are pre-determined: 70/15/15)
        test_split: Fraction for test - IGNORED (splits are pre-determined: 70/15/15)
        device: Device to load tensors on (default: 'cpu')
        cap: Maximum number of features after one-hot encoding (default: 50)
    
    Returns:
        data: dict with keys 'X_train', 'y_train', 'X_val', 'y_val', 'X_test', 'y_test'
        dataset_info: dict with metadata (includes 'task' field)
    """
    uci_dataset_map = {name: (dataset_id, task) for name, dataset_id, task in uci_datasets}
    
    if dataset_name not in uci_dataset_map:
        raise ValueError(
            f"Dataset '{dataset_name}' not found. "
            f"Available datasets: {list(uci_dataset_map.keys())}"
        )
    
    dataset_id, task = uci_dataset_map[dataset_name]
    
    X_train, y_train, X_val, y_val, X_test, y_test = get_ucidata(
        dataset_id=dataset_id,
        task=task,
        device=device,
        cap=cap
    )
    
    if task == 'regression':
        if y_train.ndim == 1:
            y_train = y_train.unsqueeze(1)
        if y_val.ndim == 1:
            y_val = y_val.unsqueeze(1)
        if y_test.ndim == 1:
            y_test = y_test.unsqueeze(1)
    elif task == 'classification':
        n_classes = len(torch.unique(y_train))
        y_train_onehot = torch.zeros(len(y_train), n_classes, dtype=torch.float64, device=device)
        y_train_onehot[torch.arange(len(y_train)), y_train] = 1.0
        y_train = y_train_onehot
        
        y_val_onehot = torch.zeros(len(y_val), n_classes, dtype=torch.float64, device=device)
        y_val_onehot[torch.arange(len(y_val)), y_val] = 1.0
        y_val = y_val_onehot
        
        y_test_onehot = torch.zeros(len(y_test), n_classes, dtype=torch.float64, device=device)
        y_test_onehot[torch.arange(len(y_test)), y_test] = 1.0
        y_test = y_test_onehot
    
    data = {
        'X_train': X_train,
        'y_train': y_train,
        'X_val': X_val,
        'y_val': y_val,
        'X_test': X_test,
        'y_test': y_test
    }
    
    dataset_info = {
        'name': dataset_name,
        'dataset_id': dataset_id,
        'n_samples': len(X_train) + len(X_val) + len(X_test),
        'n_train': len(X_train),
        'n_val': len(X_val),
        'n_test': len(X_test),
        'n_features': X_train.shape[1],
        'task': task
    }
    
    if task == 'classification':
        dataset_info['n_classes'] = n_classes
    
    return data, dataset_info
