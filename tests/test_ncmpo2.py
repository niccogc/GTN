# type: ignore
"""
Test NCMPO2 (hierarchical MPS with NTN) on MNIST.
"""

import torch
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

from testing.hierarchical_models import build_hierarchical_mps
from model.NTN import NTN
from model.losses import CrossEntropyLoss
from model.utils import CLASSIFICATION_METRICS
from model.builder import Inputs

torch.set_default_dtype(torch.float64)


def main():
    torch.manual_seed(42)
    np.random.seed(42)

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )

    train_dataset = datasets.MNIST("./data", train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST("./data", train=False, download=True, transform=transform)

    n_train = 60000
    n_val = 10000
    train_subset = Subset(train_dataset, range(n_train))
    val_subset = Subset(test_dataset, range(n_val))
    test_subset = test_dataset

    def load_and_preprocess(subset, kernel_size=7):
        import torch.nn.functional as F

        loader = DataLoader(subset, batch_size=len(subset), shuffle=False)
        images, labels = next(iter(loader))

        batch_size = images.shape[0]

        patches = F.unfold(
            images,
            kernel_size=(kernel_size, kernel_size),
            stride=(kernel_size, kernel_size),
            padding=0,
        )
        patches = patches.transpose(-2, -1)

        n_patches = patches.shape[1]
        pixels_per_patch = patches.shape[2]

        x = torch.cat(
            (patches, torch.zeros((batch_size, 1, pixels_per_patch), dtype=torch.float64)), dim=1
        )
        x = torch.cat((x, torch.zeros((batch_size, n_patches + 1, 1), dtype=torch.float64)), dim=2)

        y = torch.zeros(batch_size, 10, dtype=torch.float64)
        y.scatter_(1, labels.unsqueeze(1), 1.0)

        return x, y, n_patches, pixels_per_patch

    kernel_size = 7
    X_train, y_train, n_patches, pixels_per_patch = load_and_preprocess(train_subset, kernel_size)
    X_val, y_val, _, _ = load_and_preprocess(val_subset, kernel_size)
    X_test, y_test, _, _ = load_and_preprocess(test_subset, kernel_size)

    print(f"Data shapes: Train={X_train.shape}, Val={X_val.shape}, Test={X_test.shape}")

    tn, input_dims, output_dims = build_hierarchical_mps(
        n_sites=3,
        pixel_dim=pixels_per_patch + 1,
        patch_dim=n_patches + 1,
        rank_pixel=2,
        rank_patch=2,
        output_dim=10,
        init_strength=0.1,
    )

    print(f"\nBuilt TN: {len(tn.tensors)} tensors")
    print(f"Input dims from builder: {input_dims}")

    input_labels_3d = [(f"patch_{i}", f"pixel_{i}") for i in range(len(input_dims))]
    print(f"Input labels for Inputs class: {input_labels_3d}")

    train_loader = Inputs(
        inputs=[X_train],
        outputs=[y_train],
        outputs_labels=output_dims,
        input_labels=input_labels_3d,
        batch_dim="s",
        batch_size=100,
    )

    print(f"\nTrain loader created:")
    print(train_loader)

    val_loader = Inputs(
        inputs=[X_val],
        outputs=[y_val],
        outputs_labels=output_dims,
        input_labels=input_labels_3d,
        batch_dim="s",
        batch_size=100,
    )

    test_loader = Inputs(
        inputs=[X_test],
        outputs=[y_test],
        outputs_labels=output_dims,
        input_labels=input_labels_3d,
        batch_dim="s",
        batch_size=100,
    )

    loss_fn = CrossEntropyLoss()

    ntn = NTN(
        tn=tn,
        output_dims=output_dims,
        input_dims=[f"patch_{i}" for i in range(len(input_dims))]
        + [f"pixel_{i}" for i in range(len(input_dims))],
        loss=loss_fn,
        data_stream=train_loader,
    )

    jitter_schedule = [5 * (0.1**epoch) for epoch in range(10)]

    print("\nTraining NTN on full MNIST dataset...")
    print(f"Jitter schedule: {jitter_schedule}")

    scores_train, scores_val = ntn.fit(
        n_epochs=10,
        regularize=True,
        jitter=jitter_schedule,
        eval_metrics=CLASSIFICATION_METRICS,
        val_data=val_loader,
        verbose=True,
        patience=10,
    )

    scores_test = ntn.evaluate(CLASSIFICATION_METRICS, data_stream=test_loader)

    from model.utils import compute_quality

    print(f"\nFinal Results:")
    print(f"  Train Quality: {compute_quality(scores_train):.4f}")
    print(f"  Val Quality: {compute_quality(scores_val):.4f}")
    print(f"  Test Quality: {compute_quality(scores_test):.4f}")
    print(f"\nTest metrics: {scores_test}")


if __name__ == "__main__":
    main()
