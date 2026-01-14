# type: ignore
"""Test LMPO2 Type I on Iris dataset."""

import torch
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from model.losses import CrossEntropyLoss
from model.utils import CLASSIFICATION_METRICS, compute_quality
from model.typeI import LMPO2TypeI
from model.base import NTN_Ensemble

torch.set_default_dtype(torch.float64)


def main():
    torch.manual_seed(42)
    np.random.seed(42)

    iris = load_iris()
    X = iris.data
    y = iris.target

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    X_train = torch.tensor(X_train, dtype=torch.float64)
    X_test = torch.tensor(X_test, dtype=torch.float64)

    y_train_onehot = torch.zeros(len(y_train), 3, dtype=torch.float64)
    y_train_onehot.scatter_(1, torch.tensor(y_train).unsqueeze(1), 1.0)

    y_test_onehot = torch.zeros(len(y_test), 3, dtype=torch.float64)
    y_test_onehot.scatter_(1, torch.tensor(y_test).unsqueeze(1), 1.0)

    print(f"Iris Dataset:")
    print(f"  Train: {X_train.shape}, {y_train_onehot.shape}")
    print(f"  Test: {X_test.shape}, {y_test_onehot.shape}")

    loss_fn = CrossEntropyLoss()

    print(f"\nCreating LMPO2TypeI model builder (max_sites=4)...")
    model = LMPO2TypeI(
        max_sites=4,
        bond_dim=2,
        phys_dim=X_train.shape[1],
        reduced_dim=3,
        output_dim=3,
        init_strength=0.001,
    )

    print(f"Model created with {len(model.tns)} tensor networks")

    print(f"\nCreating NTN_Ensemble...")
    ensemble = NTN_Ensemble(
        tns=model.tns,
        input_dims_list=model.input_dims_list,
        input_labels_list=model.input_labels_list,
        output_dims=model.output_dims,
        loss=loss_fn,
        X_train=X_train,
        y_train=y_train_onehot,
        X_val=X_test,
        y_val=y_test_onehot,
        batch_size=30,
    )

    print("\nTraining...")
    scores_train, scores_val = ensemble.fit(
        n_epochs=10,
        regularize=True,
        jitter=0.1,
        eval_metrics=CLASSIFICATION_METRICS,
        verbose=True,
    )

    print(f"\nFinal Results:")
    print(f"  Train Quality: {compute_quality(scores_train):.4f}")
    print(f"  Val Quality: {compute_quality(scores_val):.4f}")


if __name__ == "__main__":
    main()
