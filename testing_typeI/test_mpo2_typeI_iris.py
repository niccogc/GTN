# type: ignore
"""
Test MPO2 Type I on Iris dataset.
"""

import torch
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from model.losses import CrossEntropyLoss
from model.utils import create_inputs, CLASSIFICATION_METRICS
from testing_typeI.mpo2_typeI import MPO2TypeI

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

    print(f"\nCreating MPO2TypeI ensemble (max_sites=4)...")
    ensemble = MPO2TypeI(
        max_sites=4,
        bond_dim=2,
        phys_dim=X_train.shape[1],
        output_dim=3,
        loss=loss_fn,
        X_data=X_train,
        y_data=y_train_onehot,
        batch_size=30,
        init_strength=0.1,
    )

    print(f"Ensemble created with {len(ensemble.ntns)} NTN instances")
    all_nodes = ensemble._get_all_trainable_nodes()
    print(f"Total trainable nodes: {len(all_nodes)}")

    test_loader = create_inputs(
        X=X_test,
        y=y_test_onehot,
        input_labels=["x0", "x1", "x2", "x3"],
        output_labels=["out"],
        batch_size=30,
        append_bias=False,
    )

    print("\nTraining...")
    scores_train, scores_test = ensemble.fit(
        n_epochs=10,
        regularize=True,
        jitter=0.01,
        eval_metrics=CLASSIFICATION_METRICS,
        val_data=test_loader,
        verbose=True,
    )

    from model.utils import compute_quality

    print(f"\nFinal Results:")
    print(f"  Train Quality: {compute_quality(scores_train):.4f}")
    print(f"  Test Quality: {compute_quality(scores_test):.4f}")


if __name__ == "__main__":
    main()
