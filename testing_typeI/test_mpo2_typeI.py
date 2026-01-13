# type: ignore
"""
Test MPO2 Type I on a simple dataset.
"""

import torch
import numpy as np
from model.losses import MSELoss
from model.utils import create_inputs, REGRESSION_METRICS
from testing_typeI.mpo2_typeI import MPO2TypeI

torch.set_default_dtype(torch.float64)


def main():
    torch.manual_seed(42)
    np.random.seed(42)

    n_samples = 200
    n_features = 10

    X = torch.randn(n_samples, n_features, dtype=torch.float64)
    y = (X[:, :3].sum(dim=1, keepdim=True) + torch.randn(n_samples, 1) * 0.1).double()

    X_train, y_train = X[:150], y[:150]
    X_val, y_val = X[150:], y[150:]

    print(f"Data: X_train={X_train.shape}, y_train={y_train.shape}")

    train_loader = create_inputs(
        X=X_train,
        y=y_train,
        input_labels=["x0", "x1", "x2"],
        output_labels=["out"],
        batch_size=50,
        append_bias=False,
    )

    val_loader = create_inputs(
        X=X_val,
        y=y_val,
        input_labels=["x0", "x1", "x2"],
        output_labels=["out"],
        batch_size=50,
        append_bias=False,
    )

    print(f"\nCreating MPO2TypeI ensemble (max_sites=3)...")
    print("  - MPO2 with 1 site")
    print("  - MPO2 with 2 sites")
    print("  - MPO2 with 3 sites")

    loss_fn = MSELoss()

    ensemble = MPO2TypeI(
        max_sites=3,
        bond_dim=2,
        phys_dim=n_features,
        output_dim=1,
        loss=loss_fn,
        data_stream=train_loader,
        init_strength=0.01,
    )

    print(f"\nEnsemble created with {len(ensemble.ntns)} NTN instances")
    print(f"Total trainable nodes: {len(ensemble._get_all_trainable_nodes())}")

    print("\nTraining...")
    scores_train, scores_val = ensemble.fit(
        n_epochs=5,
        regularize=True,
        jitter=0.01,
        eval_metrics=REGRESSION_METRICS,
        val_data=val_loader,
        verbose=True,
    )

    print(f"\nFinal Results:")
    print(f"  Train R²: {scores_train.get('quality', 0):.4f}")
    print(f"  Val R²: {scores_val.get('quality', 0):.4f}")


if __name__ == "__main__":
    main()
