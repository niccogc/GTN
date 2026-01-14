# type: ignore
"""
Test script for hierarchical MPS on MNIST.
Trains both GTN (gradient-based) and NTN (Newton-based) versions.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

from testing.hierarchical_mps import HierarchicalMPS
from testing.hierarchical_gtn import HierarchicalGTN
from testing.hierarchical_ntn import create_hierarchical_inputs
from model.NTN import NTN
from model.losses import CrossEntropyLoss
from model.utils import CLASSIFICATION_METRICS

torch.set_default_dtype(torch.float64)


def load_mnist_patches(n_train=1000, n_val=200, n_test=200, n_patches=4, seed=42):
    """
    Load MNIST and reshape into patches.

    MNIST images are 28x28. We'll reshape to (n_patches, pixels_per_patch).
    For simplicity, we'll flatten and split evenly.

    Returns:
        dict with X_train, X_val, X_test, y_train, y_val, y_test
        X shape: (n_samples, n_patches, pixels_per_patch)
        y shape: (n_samples, 10) - one-hot encoded
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )

    train_dataset = datasets.MNIST("./data", train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST("./data", train=False, download=True, transform=transform)

    train_subset = Subset(train_dataset, range(n_train))
    val_subset = Subset(train_dataset, range(n_train, n_train + n_val))
    test_subset = Subset(test_dataset, range(n_test))

    def extract_data(subset):
        loader = DataLoader(subset, batch_size=len(subset), shuffle=False)
        images, labels = next(iter(loader))

        batch_size = images.shape[0]
        images_flat = images.reshape(batch_size, -1)

        total_pixels = images_flat.shape[1]
        pixels_per_patch = total_pixels // n_patches

        images_patches = images_flat[:, : n_patches * pixels_per_patch].reshape(
            batch_size, n_patches, pixels_per_patch
        )

        labels_onehot = torch.zeros(batch_size, 10, dtype=torch.float64)
        labels_onehot.scatter_(1, labels.unsqueeze(1), 1.0)

        return images_patches, labels_onehot

    X_train, y_train = extract_data(train_subset)
    X_val, y_val = extract_data(val_subset)
    X_test, y_test = extract_data(test_subset)

    print(f"Dataset loaded:")
    print(f"  Train: {X_train.shape}, {y_train.shape}")
    print(f"  Val: {X_val.shape}, {y_val.shape}")
    print(f"  Test: {X_test.shape}, {y_test.shape}")
    print(f"  Patches: {n_patches}, Pixels per patch: {X_train.shape[2]}")

    return {
        "X_train": X_train,
        "X_val": X_val,
        "X_test": X_test,
        "y_train": y_train,
        "y_val": y_val,
        "y_test": y_test,
        "n_patches": n_patches,
        "pixels_per_patch": X_train.shape[2],
    }


def test_gtn_hierarchical(data, rank_pixel=3, rank_patch=5, n_epochs=10, lr=0.01):
    """Test GTN (gradient-based training) on hierarchical MPS."""
    print("\n" + "=" * 70)
    print("TESTING GTN (Gradient-based)")
    print("=" * 70)

    model = HierarchicalMPS(
        n_patches=data["n_patches"],
        pixels_per_patch=data["pixels_per_patch"],
        rank_pixel=rank_pixel,
        rank_patch=rank_patch,
        output_dim=10,
        init_strength=0.01,
    )

    gtn = HierarchicalGTN(model)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(gtn.parameters(), lr=lr)

    train_loader = DataLoader(
        list(zip(data["X_train"], data["y_train"])), batch_size=32, shuffle=True
    )

    for epoch in range(n_epochs):
        gtn.train()
        total_loss = 0.0

        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()

            output = gtn(batch_x)

            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * batch_x.size(0)

        avg_loss = total_loss / len(data["X_train"])

        gtn.eval()
        with torch.no_grad():
            val_output = gtn(data["X_val"])
            val_pred = val_output.argmax(dim=1)
            val_true = data["y_val"].argmax(dim=1)
            val_acc = (val_pred == val_true).float().mean().item()

        print(f"Epoch {epoch + 1}/{n_epochs} | Loss: {avg_loss:.4f} | Val Acc: {val_acc:.4f}")

    gtn.eval()
    with torch.no_grad():
        test_output = gtn(data["X_test"])
        test_pred = test_output.argmax(dim=1)
        test_true = data["y_test"].argmax(dim=1)
        test_acc = (test_pred == test_true).float().mean().item()

    print(f"\nFinal Test Accuracy: {test_acc:.4f}")
    return test_acc


def test_ntn_hierarchical(data, rank_pixel=3, rank_patch=5, n_epochs=5, jitter=0.01):
    """Test NTN (Newton-based training) on hierarchical MPS."""
    print("\n" + "=" * 70)
    print("TESTING NTN (Newton-based)")
    print("=" * 70)

    model = HierarchicalMPS(
        n_patches=data["n_patches"],
        pixels_per_patch=data["pixels_per_patch"],
        rank_pixel=rank_pixel,
        rank_patch=rank_patch,
        output_dim=10,
        init_strength=0.01,
    )

    loss_fn = CrossEntropyLoss()

    train_loader = create_hierarchical_inputs(
        X=data["X_train"], y=data["y_train"], hierarchical_model=model, batch_size=32
    )

    val_loader = create_hierarchical_inputs(
        X=data["X_val"], y=data["y_val"], hierarchical_model=model, batch_size=32
    )

    test_loader = create_hierarchical_inputs(
        X=data["X_test"], y=data["y_test"], hierarchical_model=model, batch_size=32
    )

    ntn = NTN(
        tn=model.tn,
        output_dims=model.output_dims,
        input_dims=model.input_dims,
        loss=loss_fn,
        data_stream=train_loader,
    )

    scores_train, scores_val = ntn.fit(
        n_epochs=n_epochs,
        regularize=True,
        jitter=jitter,
        eval_metrics=CLASSIFICATION_METRICS,
        val_data=val_loader,
        test_data=test_loader,
        verbose=True,
        patience=3,
        min_delta=0.001,
    )

    scores_test = ntn.evaluate(CLASSIFICATION_METRICS, data_stream=test_loader)

    print(f"\nFinal Results:")
    print(f"  Test Accuracy: {scores_test['accuracy']:.4f}")
    print(f"  Test Loss: {scores_test['loss']:.4f}")

    return scores_test["accuracy"]


def main():
    print("Loading MNIST dataset...")
    data = load_mnist_patches(n_train=1000, n_val=200, n_test=200, n_patches=4, seed=42)

    print("\n" + "=" * 70)
    print("HIERARCHICAL MPS TESTING ON MNIST")
    print("=" * 70)

    gtn_acc = test_gtn_hierarchical(data, rank_pixel=3, rank_patch=5, n_epochs=5, lr=0.01)

    ntn_acc = test_ntn_hierarchical(data, rank_pixel=3, rank_patch=5, n_epochs=3, jitter=0.01)

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"GTN Test Accuracy: {gtn_acc:.4f}")
    print(f"NTN Test Accuracy: {ntn_acc:.4f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
