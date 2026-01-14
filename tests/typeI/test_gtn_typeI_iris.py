# type: ignore
"""Test GTN Type I ensembles on Iris dataset."""

import torch
import torch.nn as nn
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from model.typeI import MPO2TypeI_GTN, LMPO2TypeI_GTN, MMPO2TypeI_GTN

torch.set_default_dtype(torch.float64)


def train_gtn_model(model, X_train, y_train, X_val, y_val, n_epochs=50, lr=0.01):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    best_val_acc = 0.0

    for epoch in range(n_epochs):
        model.train()
        optimizer.zero_grad()

        y_pred = model(X_train)
        loss = loss_fn(y_pred, y_train)

        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            train_pred = model(X_train).argmax(dim=1)
            train_acc = (train_pred == y_train).float().mean().item()

            val_pred = model(X_val).argmax(dim=1)
            val_acc = (val_pred == y_val).float().mean().item()

        is_best = val_acc > best_val_acc
        if is_best:
            best_val_acc = val_acc
            marker = " *"
        else:
            marker = ""

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(
                f"Epoch {epoch + 1:3d} | Train: loss={loss.item():.4f} acc={train_acc:.4f} | Val: acc={val_acc:.4f}{marker}"
            )

    return train_acc, val_acc


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
    y_train = torch.tensor(y_train, dtype=torch.long)
    y_test = torch.tensor(y_test, dtype=torch.long)

    print(f"Iris Dataset:")
    print(f"  Train: {X_train.shape}, {y_train.shape}")
    print(f"  Test: {X_test.shape}, {y_test.shape}")

    print("\n" + "=" * 60)
    print("Testing MPO2TypeI_GTN")
    print("=" * 60)
    model = MPO2TypeI_GTN(
        max_sites=4,
        bond_dim=4,
        phys_dim=X_train.shape[1],
        output_dim=3,
        init_strength=0.1,
    )
    print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")
    train_acc, val_acc = train_gtn_model(
        model, X_train, y_train, X_test, y_test, n_epochs=100, lr=0.01
    )
    print(f"Final: Train={train_acc:.4f}, Val={val_acc:.4f}")

    print("\n" + "=" * 60)
    print("Testing LMPO2TypeI_GTN")
    print("=" * 60)
    model = LMPO2TypeI_GTN(
        max_sites=4,
        bond_dim=4,
        phys_dim=X_train.shape[1],
        reduced_dim=3,
        output_dim=3,
        init_strength=0.1,
    )
    print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")
    train_acc, val_acc = train_gtn_model(
        model, X_train, y_train, X_test, y_test, n_epochs=100, lr=0.01
    )
    print(f"Final: Train={train_acc:.4f}, Val={val_acc:.4f}")

    print("\n" + "=" * 60)
    print("Testing MMPO2TypeI_GTN")
    print("=" * 60)
    model = MMPO2TypeI_GTN(
        max_sites=4,
        bond_dim=4,
        phys_dim=X_train.shape[1],
        output_dim=3,
        init_strength=0.1,
    )
    print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")
    train_acc, val_acc = train_gtn_model(
        model, X_train, y_train, X_test, y_test, n_epochs=100, lr=0.01
    )
    print(f"Final: Train={train_acc:.4f}, Val={val_acc:.4f}")


if __name__ == "__main__":
    main()
