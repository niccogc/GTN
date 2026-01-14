# type: ignore
"""
Test GCMPO2 (hierarchical MPS) with GTN on MNIST.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
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

    train_subset = Subset(train_dataset, range(500))
    val_subset = Subset(train_dataset, range(500, 600))
    test_subset = Subset(test_dataset, range(100))

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
        rank_pixel=2,
        rank_patch=3,
        output_dim=10,
        init_strength=0.1,
    )

    print(f"\nBuilt TN: {len(tn.tensors)} tensors, {len(input_dims)} input dims")
    print(f"Input dims: {input_dims}")
    print(f"Output dims: {output_dims}")

    print("\nTensor Network Structure:")
    for i, tensor in enumerate(tn.tensors):
        print(f"  [{i}] shape={tensor.shape}, inds={tensor.inds}, tags={tensor.tags}")

    print(f"\nInput data shape: {X_train.shape}")
    print(f"Sample input [0, :3, :5]:\n{X_train[0, :3, :5]}")

    gtn = GCMPO2(tn=tn, output_dims=["s"] + output_dims, input_dims=input_dims)

    print("\nTesting input node construction...")
    test_nodes = gtn.construct_nodes(X_train[:2])
    print(f"Created {len(test_nodes)} input nodes")
    for i, node in enumerate(test_nodes):
        print(f"  Input node {i}: shape={node.shape}, inds={node.inds}, tags={node.tags}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(gtn.parameters(), lr=0.001)

    print("\nTraining...")
    for epoch in range(5):
        optimizer.zero_grad()
        output = gtn(X_train)
        loss = criterion(output, y_train)
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            val_output = gtn(X_val)
            val_acc = (val_output.argmax(1) == y_val.argmax(1)).float().mean()

        print(f"Epoch {epoch + 1}: Loss={loss.item():.4f}, Val Acc={val_acc:.4f}")

    with torch.no_grad():
        test_output = gtn(X_test)
        test_acc = (test_output.argmax(1) == y_test.argmax(1)).float().mean()

    print(f"\nFinal Test Accuracy: {test_acc:.4f}")


if __name__ == "__main__":
    main()
