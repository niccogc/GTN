"""
Shared data preprocessing utilities for dataset loaders.
Used by load_ucirepo.py and load_from_csv.py.
"""

import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def one_hot_with_cap(X, cap=100):
    """One-hot encode categorical columns, capping total feature count.

    Args:
        X: DataFrame with mixed numeric/categorical columns
        cap: Maximum total feature count after encoding

    Returns:
        out: DataFrame with numeric cols + one-hot dummy cols
        orig_num_cols: list of original numeric column names
        dummy_cols: list of one-hot dummy column names
    """
    num_X = X.select_dtypes(exclude=["object", "category"])
    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()

    available = cap - num_X.shape[1]
    if available <= 0 or len(cat_cols) == 0:
        out = num_X.copy()
        return out, num_X.columns.tolist(), []

    class_counts = X[cat_cols].nunique(dropna=True)
    total_needed = int(class_counts.sum())
    to_drop = []

    if total_needed > available:
        for col, cnt in class_counts.sort_values(ascending=False).items():
            if total_needed <= available:
                break
            to_drop.append(col)
            total_needed -= int(cnt)

    keep_cols = [c for c in cat_cols if c not in to_drop]

    if keep_cols:
        dummies = pd.get_dummies(X[keep_cols], prefix=keep_cols, dummy_na=True, dtype=int)
        out = pd.concat([num_X, dummies], axis=1)
        dummy_cols = dummies.columns.tolist()
    else:
        out = num_X.copy()
        dummy_cols = []

    if out.shape[1] > cap:
        all_num_cols = num_X.columns.tolist()
        room = max(cap - len(all_num_cols), 0)
        trimmed_dummy_cols = dummy_cols[:room]
        out = pd.concat([num_X, out[trimmed_dummy_cols]], axis=1)
        dummy_cols = trimmed_dummy_cols

    return out, num_X.columns.tolist(), dummy_cols


def split_data(X, y, val_size=0.15, test_size=0.15, random_state=42):
    """Split into train/val/test with fixed seed for reproducibility.

    Returns:
        X_train, X_val, X_test, y_train, y_val, y_test: DataFrames/Series
    """
    test_val_size = val_size + test_size
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=test_val_size, random_state=random_state
    )
    # Split temp into val and test (equal halves of the 30%)
    relative_test_size = test_size / test_val_size
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=relative_test_size, random_state=random_state
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def scale_dataframes(
    X_train, X_val, X_test, num_cols=None
):
    """Fit StandardScaler on training data and transform all splits.

    Args:
        X_train, X_val, X_test: DataFrames
        num_cols: Columns to scale. If None, scales ALL columns.

    Returns:
        Transformed X_train, X_val, X_test DataFrames
    """
    if num_cols is None:
        num_cols = X_train.columns.tolist()

    if len(num_cols) == 0:
        return X_train, X_val, X_test

    scaler = StandardScaler()
    scaler.fit(X_train[num_cols])

    X_train = X_train.copy()
    X_val = X_val.copy()
    X_test = X_test.copy()

    X_train[num_cols] = X_train[num_cols].astype(float)
    X_val[num_cols] = X_val[num_cols].astype(float)
    X_test[num_cols] = X_test[num_cols].astype(float)

    X_train.loc[:, num_cols] = scaler.transform(X_train[num_cols])
    X_val.loc[:, num_cols] = scaler.transform(X_val[num_cols])
    X_test.loc[:, num_cols] = scaler.transform(X_test[num_cols])

    return X_train, X_val, X_test


def to_tensors(
    X_train, X_val, X_test, y_train, y_val, y_test, task, device="cpu"
):
    """Convert pandas DataFrames/Series to torch tensors.

    Args:
        task: "regression" → y is float64; "classification" → y is long
        device: Target device

    Returns:
        X_train, y_train, X_val, y_val, X_test, y_test: torch tensors
    """
    y_dtype = torch.float64 if task == "regression" else torch.long

    X_train_t = torch.tensor(X_train.values, dtype=torch.float64, device=device)
    y_train_t = torch.tensor(y_train.values, dtype=y_dtype, device=device)
    if task == "regression" and y_train_t.ndim == 1:
        y_train_t = y_train_t.unsqueeze(1)

    X_val_t = torch.tensor(X_val.values, dtype=torch.float64, device=device)
    y_val_t = torch.tensor(y_val.values, dtype=y_dtype, device=device)
    if task == "regression" and y_val_t.ndim == 1:
        y_val_t = y_val_t.unsqueeze(1)

    X_test_t = torch.tensor(X_test.values, dtype=torch.float64, device=device)
    y_test_t = torch.tensor(y_test.values, dtype=y_dtype, device=device)
    if task == "regression" and y_test_t.ndim == 1:
        y_test_t = y_test_t.unsqueeze(1)

    return X_train_t, y_train_t, X_val_t, y_val_t, X_test_t, y_test_t


def process_classification_targets(y_train, y_val, y_test, device="cpu"):
    """Convert classification targets to one-hot encoded tensors.

    Returns:
        y_train, y_val, y_test: one-hot float64 tensors
        n_classes: int
    """
    n_classes = len(torch.unique(y_train))

    def to_onehot(y):
        oh = torch.zeros(len(y), n_classes, dtype=torch.float64, device=device)
        oh[torch.arange(len(y)), y] = 1.0
        return oh

    return to_onehot(y_train), to_onehot(y_val), to_onehot(y_test), n_classes
