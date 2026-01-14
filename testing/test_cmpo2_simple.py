# type: ignore
"""
Test CMPO2 models on simple synthetic data (no MNIST to avoid memory issues).
"""

import torch
import numpy as np

from testing.CMPO2_models import CMPO2, CMPO2_GTN, CMPO2TypeI, CMPO2TypeI_GTN, CMPO2TypeI_NTN
from model.base import NTN
from model.losses import CrossEntropyLoss
from model.utils import CLASSIFICATION_METRICS, compute_quality
from model.builder import Inputs

torch.set_default_dtype(torch.float64)


def create_synthetic_data(n_samples, n_patches, pixels_per_patch, n_classes):
    X = torch.randn(n_samples, n_patches, pixels_per_patch, dtype=torch.float64)
    y_labels = torch.randint(0, n_classes, (n_samples,))
    y = torch.zeros(n_samples, n_classes, dtype=torch.float64)
    y.scatter_(1, y_labels.unsqueeze(1), 1.0)
    return X, y


def test_cmpo2_ntn():
    print("=" * 60)
    print("Testing CMPO2 with NTN")
    print("=" * 60)

    torch.manual_seed(42)
    np.random.seed(42)

    n_samples = 100
    n_patches = 5
    pixels_per_patch = 10
    n_classes = 3

    X_train, y_train = create_synthetic_data(60, n_patches, pixels_per_patch, n_classes)
    X_val, y_val = create_synthetic_data(20, n_patches, pixels_per_patch, n_classes)
    X_test, y_test = create_synthetic_data(20, n_patches, pixels_per_patch, n_classes)

    print(f"Data: X_train={X_train.shape}, y_train={y_train.shape}")

    model = CMPO2(
        n_sites=3,
        pixel_dim=pixels_per_patch,
        patch_dim=n_patches,
        rank_pixel=2,
        rank_patch=2,
        output_dim=n_classes,
        init_strength=0.1,
    )

    print(f"Model created: {len(model.tn.tensors)} tensors")
    print(f"Input labels: {model.input_labels}")
    print(f"Input dims: {model.input_dims}")

    train_loader = Inputs(
        inputs=[X_train],
        outputs=[y_train],
        outputs_labels=model.output_dims,
        input_labels=model.input_labels,
        batch_dim="s",
        batch_size=30,
    )

    val_loader = Inputs(
        inputs=[X_val],
        outputs=[y_val],
        outputs_labels=model.output_dims,
        input_labels=model.input_labels,
        batch_dim="s",
        batch_size=30,
    )

    loss_fn = CrossEntropyLoss()

    all_input_dims = []
    for patch_label, pixel_label in model.input_labels:
        all_input_dims.extend([patch_label, pixel_label])

    ntn = NTN(
        tn=model.tn,
        output_dims=model.output_dims,
        input_dims=all_input_dims,
        loss=loss_fn,
        data_stream=train_loader,
    )

    print("\nTraining...")
    scores_train, scores_val = ntn.fit(
        n_epochs=5,
        regularize=True,
        jitter=0.1,
        eval_metrics=CLASSIFICATION_METRICS,
        val_data=val_loader,
        verbose=True,
    )

    print(
        f"\nFinal: Train={compute_quality(scores_train):.4f}, Val={compute_quality(scores_val):.4f}"
    )


def test_cmpo2_gtn():
    print("\n" + "=" * 60)
    print("Testing CMPO2 with GTN")
    print("=" * 60)

    torch.manual_seed(42)
    np.random.seed(42)

    n_patches = 5
    pixels_per_patch = 10
    n_classes = 3

    X_train, y_train = create_synthetic_data(60, n_patches, pixels_per_patch, n_classes)
    X_val, y_val = create_synthetic_data(20, n_patches, pixels_per_patch, n_classes)

    print(f"Data: X_train={X_train.shape}, y_train={y_train.shape}")

    model = CMPO2(
        n_sites=3,
        pixel_dim=pixels_per_patch,
        patch_dim=n_patches,
        rank_pixel=2,
        rank_patch=2,
        output_dim=n_classes,
        init_strength=0.1,
    )

    gtn = CMPO2_GTN(model)

    print(f"GTN created with {sum(p.numel() for p in gtn.parameters())} parameters")

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(gtn.parameters(), lr=0.01)

    print("\nTraining...")
    for epoch in range(10):
        optimizer.zero_grad()
        output = gtn(X_train)
        loss = criterion(output, y_train)
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            train_acc = (output.argmax(1) == y_train.argmax(1)).float().mean()
            val_output = gtn(X_val)
            val_acc = (val_output.argmax(1) == y_val.argmax(1)).float().mean()

        if (epoch + 1) % 2 == 0 or epoch == 0:
            print(
                f"Epoch {epoch + 1:2d} | Loss={loss.item():.4f} | Train={train_acc:.4f} | Val={val_acc:.4f}"
            )

    print(f"\nFinal: Train={train_acc:.4f}, Val={val_acc:.4f}")


def test_cmpo2_typeI_gtn():
    print("\n" + "=" * 60)
    print("Testing CMPO2TypeI with GTN")
    print("=" * 60)

    torch.manual_seed(42)
    np.random.seed(42)

    n_patches = 5
    pixels_per_patch = 10
    n_classes = 3

    X_train, y_train = create_synthetic_data(60, n_patches, pixels_per_patch, n_classes)
    X_val, y_val = create_synthetic_data(20, n_patches, pixels_per_patch, n_classes)

    print(f"Data: X_train={X_train.shape}, y_train={y_train.shape}")

    model = CMPO2TypeI(
        max_sites=3,
        pixel_dim=pixels_per_patch,
        patch_dim=n_patches,
        rank_pixel=2,
        rank_patch=2,
        output_dim=n_classes,
        init_strength=0.1,
    )

    print(f"Model created with {len(model.tns)} tensor networks")

    gtn = CMPO2TypeI_GTN(model)

    print(f"GTN ensemble with {sum(p.numel() for p in gtn.parameters())} parameters")

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(gtn.parameters(), lr=0.01)

    print("\nTraining...")
    for epoch in range(10):
        optimizer.zero_grad()
        output = gtn(X_train)
        loss = criterion(output, y_train)
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            train_acc = (output.argmax(1) == y_train.argmax(1)).float().mean()
            val_output = gtn(X_val)
            val_acc = (val_output.argmax(1) == y_val.argmax(1)).float().mean()

        if (epoch + 1) % 2 == 0 or epoch == 0:
            print(
                f"Epoch {epoch + 1:2d} | Loss={loss.item():.4f} | Train={train_acc:.4f} | Val={val_acc:.4f}"
            )

    print(f"\nFinal: Train={train_acc:.4f}, Val={val_acc:.4f}")


def test_cmpo2_typeI_ntn():
    print("\n" + "=" * 60)
    print("Testing CMPO2TypeI with CMPO2TypeI_NTN")
    print("=" * 60)

    torch.manual_seed(42)
    np.random.seed(42)

    n_patches = 5
    pixels_per_patch = 10
    n_classes = 3

    X_train, y_train = create_synthetic_data(60, n_patches, pixels_per_patch, n_classes)
    X_val, y_val = create_synthetic_data(20, n_patches, pixels_per_patch, n_classes)

    print(f"Data: X_train={X_train.shape}, y_train={y_train.shape}")

    model = CMPO2TypeI(
        max_sites=3,
        pixel_dim=pixels_per_patch,
        patch_dim=n_patches,
        rank_pixel=2,
        rank_patch=2,
        output_dim=n_classes,
        init_strength=0.1,
    )

    print(f"Model created with {len(model.tns)} tensor networks")

    loss_fn = CrossEntropyLoss()

    ensemble = CMPO2TypeI_NTN(
        cmpo2_typeI=model,
        loss=loss_fn,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        batch_size=30,
    )

    print("\nTraining...")
    scores_train, scores_val = ensemble.fit(
        n_epochs=5,
        regularize=True,
        jitter=0.1,
        eval_metrics=CLASSIFICATION_METRICS,
        verbose=True,
    )

    print(
        f"\nFinal: Train={compute_quality(scores_train):.4f}, Val={compute_quality(scores_val):.4f}"
    )


if __name__ == "__main__":
    test_cmpo2_ntn()
    test_cmpo2_gtn()
    test_cmpo2_typeI_gtn()
    test_cmpo2_typeI_ntn()
