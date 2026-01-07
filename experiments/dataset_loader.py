# type: ignore
"""
Dataset loader utility for production training scripts.
"""
import torch
import numpy as np
from sklearn.preprocessing import StandardScaler


def load_dataset(dataset_name, n_samples=None, seed=0, val_split=0.2, test_split=0.2):
    """
    Load a dataset by name and return standardized features and targets with train/val/test splits.
    
    Args:
        dataset_name: Name of dataset to load
        n_samples: Number of samples to use (None = use all)
        seed: Random seed for sampling and splitting
        val_split: Fraction of data for validation (default: 0.2)
        test_split: Fraction of data for test (default: 0.2)
    
    Returns:
        data: dict with keys 'X_train', 'y_train', 'X_val', 'y_val', 'X_test', 'y_test'
        dataset_info: dict with metadata (includes 'task' field)
    """
    
    if dataset_name == "california_housing":
        from sklearn import datasets
        from sklearn.model_selection import train_test_split
        
        housing = datasets.fetch_california_housing()
        X = housing.data
        y = housing.target
        
        # Sample subset if requested
        if n_samples is not None and n_samples < len(X):
            np.random.seed(seed)
            indices = np.random.choice(len(X), n_samples, replace=False)
            X = X[indices]
            y = y[indices]
        
        # Standardize features
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        
        # Train/val/test split
        # First split: train+val vs test
        X_trainval, X_test, y_trainval, y_test = train_test_split(
            X, y, test_size=test_split, random_state=seed
        )
        
        # Second split: train vs val
        val_size_adjusted = val_split / (1 - test_split)  # Adjust val proportion
        X_train, X_val, y_train, y_val = train_test_split(
            X_trainval, y_trainval, test_size=val_size_adjusted, random_state=seed
        )
        
        # Convert to torch
        data = {
            'X_train': torch.tensor(X_train, dtype=torch.float32),
            'y_train': torch.tensor(y_train, dtype=torch.float32).unsqueeze(1),
            'X_val': torch.tensor(X_val, dtype=torch.float32),
            'y_val': torch.tensor(y_val, dtype=torch.float32).unsqueeze(1),
            'X_test': torch.tensor(X_test, dtype=torch.float32),
            'y_test': torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)
        }
        
        dataset_info = {
            'name': 'california_housing',
            'n_samples': len(X),
            'n_train': len(X_train),
            'n_val': len(X_val),
            'n_test': len(X_test),
            'n_features': X.shape[1],
            'task': 'regression'
        }
        
        return data, dataset_info
    
    elif dataset_name == "iris":
        from sklearn import datasets
        from sklearn.model_selection import train_test_split
        
        iris = datasets.load_iris()
        X = iris.data
        y = iris.target
        
        # Sample subset if requested
        if n_samples is not None and n_samples < len(X):
            np.random.seed(seed)
            indices = np.random.choice(len(X), n_samples, replace=False)
            X = X[indices]
            y = y[indices]
        
        # Standardize features
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        
        # Train/val/test split
        # First split: train+val vs test
        X_trainval, X_test, y_trainval, y_test = train_test_split(
            X, y, test_size=test_split, random_state=seed, stratify=y
        )
        
        # Second split: train vs val
        val_size_adjusted = val_split / (1 - test_split)
        X_train, X_val, y_train, y_val = train_test_split(
            X_trainval, y_trainval, test_size=val_size_adjusted, random_state=seed, stratify=y_trainval
        )
        
        # Convert labels to one-hot encoding for classification
        n_classes = len(np.unique(y))
        
        def to_onehot(labels, n_classes):
            onehot = np.zeros((len(labels), n_classes))
            onehot[np.arange(len(labels)), labels] = 1
            return onehot
        
        # Convert to torch
        data = {
            'X_train': torch.tensor(X_train, dtype=torch.float32),
            'y_train': torch.tensor(to_onehot(y_train, n_classes), dtype=torch.float32),
            'X_val': torch.tensor(X_val, dtype=torch.float32),
            'y_val': torch.tensor(to_onehot(y_val, n_classes), dtype=torch.float32),
            'X_test': torch.tensor(X_test, dtype=torch.float32),
            'y_test': torch.tensor(to_onehot(y_test, n_classes), dtype=torch.float32)
        }
        
        dataset_info = {
            'name': 'iris',
            'n_samples': len(X),
            'n_train': len(X_train),
            'n_val': len(X_val),
            'n_test': len(X_test),
            'n_features': X.shape[1],
            'n_classes': n_classes,
            'task': 'classification'
        }
        
        return data, dataset_info
    
    else:
        raise NotImplementedError(
            f"Dataset '{dataset_name}' not implemented. "
            f"Please add loading logic in dataset_loader.py"
        )
