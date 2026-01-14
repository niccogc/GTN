# type: ignore
"""
Test MPO2 Type I on a simple dataset.
"""

import torch
import numpy as np
from model.losses import MSELoss
from model.utils import REGRESSION_METRICS
from model.typeI import MPO2TypeI
from model.base import NTN_Ensemble

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

    print(f"\nCreating MPO2TypeI model builder (max_sites=3)...")
    print("  - MPO2 with 1 site")
    print("  - MPO2 with 2 sites")
    print("  - MPO2 with 3 sites")

    loss_fn = MSELoss()

    model = MPO2TypeI(
        max_sites=3,
        bond_dim=2,
        phys_dim=n_features,
        output_dim=1,
        init_strength=0.01,
    )

    print(f"\nModel created with {len(model.tns)} tensor networks")

    print(f"\nCreating NTN_Ensemble...")
    ensemble = NTN_Ensemble(
        tns=model.tns,
        input_dims_list=model.input_dims_list,
        input_labels_list=model.input_labels_list,
        output_dims=model.output_dims,
        loss=loss_fn,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        batch_size=50,
    )

    print("\nTraining...")
    scores_train, scores_val = ensemble.fit(
        n_epochs=5,
        regularize=True,
        jitter=0.01,
        eval_metrics=REGRESSION_METRICS,
        verbose=True,
    )

    from model.utils import compute_quality

    print(f"\nFinal Results:")
    print(f"  Train R²: {compute_quality(scores_train):.4f}")
    print(f"  Val R²: {compute_quality(scores_val):.4f}")


if __name__ == "__main__":
    main()
