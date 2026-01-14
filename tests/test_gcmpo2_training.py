# type: ignore
"""
Train GCMPO2 for 300 epochs and plot results.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

from testing.hierarchical_models import build_hierarchical_mps, GCMPO2

torch.set_default_dtype(torch.float64)


def main():
    torch.manual_seed(42)
    np.random.seed(42)

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )

    train_dataset = datasets.MNIST("./data", train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST("./data", train=False, download=True, transform=transform)

    train_subset = Subset(train_dataset, range(1000))
    val_subset = Subset(train_dataset, range(1000, 1200))
    test_subset = Subset(test_dataset, range(500))

    def load_and_preprocess(subset, n_patches=4):
        loader = DataLoader(subset, batch_size=len(subset), shuffle=False)
        images, labels = next(iter(loader))

        batch_size = images.shape[0]
        images_flat = images.reshape(batch_size, -1)

        pixels_per_patch = images_flat.shape[1] // n_patches
        images_patches = images_flat[:, : n_patches * pixels_per_patch].reshape(
            batch_size, n_patches, pixels_per_patch
        )

        x = torch.cat(
            (images_patches, torch.zeros((batch_size, 1, pixels_per_patch), dtype=torch.float64)),
            dim=1,
        )
        x = torch.cat((x, torch.zeros((batch_size, n_patches + 1, 1), dtype=torch.float64)), dim=2)

        y = torch.zeros(batch_size, 10, dtype=torch.float64)
        y.scatter_(1, labels.unsqueeze(1), 1.0)

        return x, y, n_patches, pixels_per_patch

    X_train, y_train, n_patches, pixels_per_patch = load_and_preprocess(train_subset)
    X_val, y_val, _, _ = load_and_preprocess(val_subset, n_patches)
    X_test, y_test, _, _ = load_and_preprocess(test_subset, n_patches)

    print(f"Data shapes: Train={X_train.shape}, Val={X_val.shape}, Test={X_test.shape}")

    tn, input_dims, output_dims = build_hierarchical_mps(
        n_sites=3,
        pixel_dim=pixels_per_patch + 1,
        patch_dim=n_patches + 1,
        rank_pixel=4,
        rank_patch=4,
        output_dim=10,
        init_strength=0.1,
    )

    print(f"Built TN: {len(tn.tensors)} tensors, rank_pixel=4, rank_patch=4")

    gtn = GCMPO2(tn=tn, output_dims=["s"] + output_dims, input_dims=input_dims)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(gtn.parameters(), lr=0.001)

    train_losses = []
    val_accs = []
    train_accs = []

    print("\nTraining for 300 epochs...")
    for epoch in range(300):
        optimizer.zero_grad()
        output = gtn(X_train)
        loss = criterion(output, y_train)
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            train_acc = (output.argmax(1) == y_train.argmax(1)).float().mean().item()
            val_output = gtn(X_val)
            val_acc = (val_output.argmax(1) == y_val.argmax(1)).float().mean().item()

        train_losses.append(loss.item())
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        if (epoch + 1) % 50 == 0:
            print(
                f"Epoch {epoch + 1}: Loss={loss.item():.4f}, Train Acc={train_acc:.4f}, Val Acc={val_acc:.4f}"
            )

    with torch.no_grad():
        test_output = gtn(X_test)
        test_acc = (test_output.argmax(1) == y_test.argmax(1)).float().mean().item()

    print(f"\nFinal Results:")
    print(f"  Train Accuracy: {train_accs[-1]:.4f}")
    print(f"  Val Accuracy: {val_accs[-1]:.4f}")
    print(f"  Test Accuracy: {test_acc:.4f}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(train_losses, label="Train Loss", linewidth=2)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training Loss")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    ax2.plot(train_accs, label="Train Accuracy", linewidth=2)
    ax2.plot(val_accs, label="Val Accuracy", linewidth=2)
    ax2.axhline(y=test_acc, color="r", linestyle="--", label=f"Test Acc = {test_acc:.4f}")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Accuracy over Training")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    plt.tight_layout()
    plt.savefig("./testing/gcmpo2_training.png", dpi=150)
    print("\nPlot saved to ./testing/gcmpo2_training.png")


if __name__ == "__main__":
    main()
