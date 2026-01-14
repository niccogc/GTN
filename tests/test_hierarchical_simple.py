# type: ignore
"""
Simple test for hierarchical MPS (GCMPO2) on MNIST - GTN only.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

from testing.hierarchical_models import build_hierarchical_mps, GCMPO2

torch.set_default_dtype(torch.float64)


def load_mnist_simple(n_train=500, n_val=100, n_test=100, n_patches=4, seed=42):
    """Load MNIST and preprocess with one-hot encoding for patches/pixels."""
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

    def preprocess_data(subset):
        loader = DataLoader(subset, batch_size=len(subset), shuffle=False)
        images, labels = next(iter(loader))

        batch_size = images.shape[0]
        images_flat = images.reshape(batch_size, -1)

        total_pixels = images_flat.shape[1]
        pixels_per_patch = total_pixels // n_patches

        images_patches = images_flat[:, : n_patches * pixels_per_patch].reshape(
            batch_size, n_patches, pixels_per_patch
        )

        x = torch.cat(
            (images_patches, torch.zeros((batch_size, 1, pixels_per_patch), dtype=torch.float64)),
            dim=1,
        )
        x = torch.cat((x, torch.zeros((batch_size, n_patches + 1, 1), dtype=torch.float64)), dim=2)

        labels_onehot = torch.zeros(batch_size, 10, dtype=torch.float64)
        labels_onehot.scatter_(1, labels.unsqueeze(1), 1.0)

        return x, labels_onehot

    X_train, y_train = preprocess_data(train_subset)
    X_val, y_val = preprocess_data(val_subset)
    X_test, y_test = preprocess_data(test_subset)

    pixels_per_patch = X_train.shape[2] - 1

    print(f"Dataset loaded:")
    print(f"  Train: {X_train.shape}")
    print(f"  Val: {X_val.shape}")
    print(f"  Test: {X_test.shape}")
    print(f"  Patches: {n_patches}, Pixels per patch: {pixels_per_patch}")

    return {
        "X_train": X_train,
        "X_val": X_val,
        "X_test": X_test,
        "y_train": y_train,
        "y_val": y_val,
        "y_test": y_test,
        "n_patches": n_patches,
        "pixels_per_patch": pixels_per_patch,
    }


def test_gtn(data, rank_pixel=2, rank_patch=3, n_epochs=3):
    """Test GCMPO2 with GTN."""
    print("\n" + "=" * 60)
    print("GCMPO2 GTN TEST")
    print("=" * 60)

    tn, input_dims, output_dims = build_hierarchical_mps(
        n_patches=data["n_patches"],
        pixels_per_patch=data["pixels_per_patch"],
        rank_pixel=rank_pixel,
        rank_patch=rank_patch,
        output_dim=10,
        init_strength=0.01,
    )

    gtn = GCMPO2(tn=tn, output_dims=["s"] + output_dims, input_dims=input_dims)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(gtn.parameters(), lr=0.01)

    for epoch in range(n_epochs):
        optimizer.zero_grad()
        output = gtn(data["X_train"])
        loss = criterion(output, data["y_train"])
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            val_out = gtn(data["X_val"])
            val_acc = (val_out.argmax(1) == data["y_val"].argmax(1)).float().mean()

        print(f"Epoch {epoch + 1}: Loss={loss.item():.4f}, Val Acc={val_acc:.4f}")

    with torch.no_grad():
        test_out = gtn(data["X_test"])
        test_acc = (test_out.argmax(1) == data["y_test"].argmax(1)).float().mean()

    print(f"Test Accuracy: {test_acc:.4f}")


def main():
    print("Loading MNIST...")
    data = load_mnist_simple(n_train=500, n_val=100, n_test=100, n_patches=4)

    test_gtn(data, rank_pixel=2, rank_patch=3, n_epochs=3)


if __name__ == "__main__":
    main()
