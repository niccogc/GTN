# type: ignore
"""
Image dataset loader for CMPO2 experiments.
Loads MNIST, FASHION_MNIST, CIFAR10, CIFAR100 and preprocesses into
(batch, n_patches, pixels_per_patch) format for CMPO2 models.
"""

import os
import torch
import numpy as np
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import datasets, transforms
from typing import Tuple, Dict, Any, Optional

torch.set_default_dtype(torch.float64)


def get_image_transforms(dataset_name: str) -> transforms.Compose:
    if dataset_name in ["MNIST", "FASHION_MNIST"]:
        return transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        )
    elif dataset_name == "CIFAR10":
        return transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
            ]
        )
    elif dataset_name == "CIFAR100":
        return transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
            ]
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


def get_dataset_info(dataset_name: str) -> Dict[str, Any]:
    info = {
        "MNIST": {
            "image_size": 28,
            "channels": 1,
            "n_classes": 10,
            "total_pixels": 784,
        },
        "FASHION_MNIST": {
            "image_size": 28,
            "channels": 1,
            "n_classes": 10,
            "total_pixels": 784,
        },
        "CIFAR10": {
            "image_size": 32,
            "channels": 3,
            "n_classes": 10,
            "total_pixels": 3072,
        },
        "CIFAR100": {
            "image_size": 32,
            "channels": 3,
            "n_classes": 100,
            "total_pixels": 3072,
        },
    }
    if dataset_name not in info:
        raise ValueError(f"Unknown dataset: {dataset_name}. Available: {list(info.keys())}")
    return info[dataset_name]


def preprocess_for_cmpo2(
    images: torch.Tensor,
    labels: torch.Tensor,
    n_patches: int,
    pixels_per_patch: int,
    n_classes: int,
    device: str = "cpu",
) -> Tuple[torch.Tensor, torch.Tensor]:
    batch_size = images.shape[0]
    images_flat = images.reshape(batch_size, -1).to(torch.float64)

    total_needed = n_patches * pixels_per_patch
    images_flat = images_flat[:, :total_needed]

    X = images_flat.reshape(batch_size, n_patches, pixels_per_patch)

    y = torch.zeros(batch_size, n_classes, dtype=torch.float64)
    y.scatter_(1, labels.unsqueeze(1), 1.0)

    return X.to(device), y.to(device)


def preprocess_for_cmpo3(
    images: torch.Tensor,
    labels: torch.Tensor,
    n_patches: int,
    pixels_per_patch: int,
    n_classes: int,
    device: str = "cpu",
) -> Tuple[torch.Tensor, torch.Tensor]:
    batch_size = images.shape[0]
    n_channels = images.shape[1]

    images = images.to(torch.float64)
    images = images.permute(0, 2, 3, 1).reshape(batch_size, -1, n_channels)

    total_needed = n_patches * pixels_per_patch
    images = images[:, :total_needed, :]

    X = images.reshape(batch_size, n_patches, pixels_per_patch, n_channels)

    y = torch.zeros(batch_size, n_classes, dtype=torch.float64)
    y.scatter_(1, labels.unsqueeze(1), 1.0)

    return X.to(device), y.to(device)


def load_image_dataset(
    dataset_name: str,
    n_patches: int = 4,
    pixels_per_patch: Optional[int] = None,
    n_train: Optional[int] = None,
    n_val: Optional[int] = None,
    n_test: Optional[int] = None,
    val_split: float = 0.15,
    device: str = "cpu",
    data_dir: Optional[str] = None,
    model_type: str = "cmpo2",
) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
    if data_dir is None:
        data_dir = os.path.join(os.path.expanduser("~"), "data")
    transform = get_image_transforms(dataset_name)
    info = get_dataset_info(dataset_name)

    if dataset_name == "MNIST":
        train_dataset = datasets.MNIST(data_dir, train=True, download=True, transform=transform)
        test_dataset = datasets.MNIST(data_dir, train=False, download=True, transform=transform)
    elif dataset_name == "FASHION_MNIST":
        train_dataset = datasets.FashionMNIST(
            data_dir, train=True, download=True, transform=transform
        )
        test_dataset = datasets.FashionMNIST(
            data_dir, train=False, download=True, transform=transform
        )
    elif dataset_name == "CIFAR10":
        train_dataset = datasets.CIFAR10(data_dir, train=True, download=True, transform=transform)
        test_dataset = datasets.CIFAR10(data_dir, train=False, download=True, transform=transform)
    elif dataset_name == "CIFAR100":
        train_dataset = datasets.CIFAR100(data_dir, train=True, download=True, transform=transform)
        test_dataset = datasets.CIFAR100(data_dir, train=False, download=True, transform=transform)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    generator = torch.Generator().manual_seed(42)

    total_train = len(train_dataset)
    if n_val is None:
        n_val_actual = int(total_train * val_split)
    else:
        n_val_actual = n_val

    n_train_actual = total_train - n_val_actual

    train_subset, val_subset = random_split(
        train_dataset, [n_train_actual, n_val_actual], generator=generator
    )

    if n_train is not None and n_train < len(train_subset):
        train_subset = Subset(train_subset, range(n_train))
    if n_val is not None and n_val < len(val_subset):
        val_subset = Subset(val_subset, range(n_val))
    if n_test is not None and n_test < len(test_dataset):
        test_dataset = Subset(test_dataset, range(n_test))

    def load_subset(subset):
        loader = DataLoader(subset, batch_size=len(subset), shuffle=False)
        images, labels = next(iter(loader))
        return images, labels

    train_images, train_labels = load_subset(train_subset)
    val_images, val_labels = load_subset(val_subset)
    test_images, test_labels = load_subset(
        test_dataset if n_test is None else Subset(test_dataset, range(len(test_dataset)))
    )

    n_classes = info["n_classes"]
    n_channels = info["channels"]
    spatial_pixels = info["image_size"] * info["image_size"]

    if pixels_per_patch is None:
        pixels_per_patch = spatial_pixels // n_patches

    if model_type == "cmpo3":
        X_train, y_train = preprocess_for_cmpo3(
            train_images, train_labels, n_patches, pixels_per_patch, n_classes, device
        )
        X_val, y_val = preprocess_for_cmpo3(
            val_images, val_labels, n_patches, pixels_per_patch, n_classes, device
        )
        X_test, y_test = preprocess_for_cmpo3(
            test_images, test_labels, n_patches, pixels_per_patch, n_classes, device
        )
    else:
        X_train, y_train = preprocess_for_cmpo2(
            train_images, train_labels, n_patches, pixels_per_patch, n_classes, device
        )
        X_val, y_val = preprocess_for_cmpo2(
            val_images, val_labels, n_patches, pixels_per_patch, n_classes, device
        )
        X_test, y_test = preprocess_for_cmpo2(
            test_images, test_labels, n_patches, pixels_per_patch, n_classes, device
        )

    data = {
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
        "X_test": X_test,
        "y_test": y_test,
    }

    dataset_info = {
        "name": dataset_name,
        "task": "classification",
        "model_type": model_type,
        "n_classes": n_classes,
        "n_train": len(X_train),
        "n_val": len(X_val),
        "n_test": len(X_test),
        "image_size": info["image_size"],
        "n_channels": n_channels,
        "n_patches": n_patches,
        "pixels_per_patch": pixels_per_patch,
        "patch_dim": n_patches,
        "pixel_dim": pixels_per_patch,
        "channel_dim": n_channels,
    }

    return data, dataset_info


def get_valid_patch_configs(dataset_name: str):
    info = get_dataset_info(dataset_name)
    spatial = info["image_size"] * info["image_size"]
    configs = []
    for n_patches in range(1, spatial + 1):
        if spatial % n_patches == 0:
            pixels_per_patch = spatial // n_patches
            configs.append((n_patches, pixels_per_patch))
    return configs


if __name__ == "__main__":
    for ds in ["MNIST", "CIFAR10"]:
        info = get_dataset_info(ds)
        spatial = info["image_size"] * info["image_size"]
        print(f"\n{ds} (spatial={spatial}, channels={info['channels']}):")
        print("  n_patches x pixels_per_patch:")
        for n_p, ppp in get_valid_patch_configs(ds):
            if n_p <= 32 and ppp >= 4:
                print(f"    {n_p} x {ppp}")
