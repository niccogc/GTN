#!/usr/bin/env python3
"""Test MMPO2TypeI_GTN device handling."""

import torch
import sys

sys.path.insert(0, ".")

from model.typeI import MMPO2TypeI_GTN


def test_device_handling():
    print("Creating MMPO2TypeI_GTN model...")
    model = MMPO2TypeI_GTN(
        max_sites=3,
        bond_dim=2,
        phys_dim=5,
        output_dim=3,
    )

    print("\nBefore .to(device):")
    for i, m in enumerate(model.models):
        print(f"  Model {i}:")
        print(f"    torch_params devices: {set(p.device.type for p in m.torch_params.values())}")
        print(
            f"    not_trainable_params devices: {set(v.device.type for v in m.not_trainable_params.values())}"
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nMoving to device: {device}")
    model = model.to(device)

    print("\nAfter .to(device):")
    for i, m in enumerate(model.models):
        print(f"  Model {i}:")
        print(f"    torch_params devices: {set(p.device.type for p in m.torch_params.values())}")
        print(
            f"    not_trainable_params devices: {set(v.device.type for v in m.not_trainable_params.values())}"
        )

    print("\nTesting forward pass...")
    x = torch.randn(4, 5).to(device)
    print(f"  Input device: {x.device}")

    try:
        out = model(x)
        print(f"  Output device: {out.device}")
        print(f"  Output shape: {out.shape}")
        print("\n✓ SUCCESS: Forward pass completed without errors!")
        return True
    except Exception as e:
        print(f"\n✗ FAILED: {e}")
        return False


if __name__ == "__main__":
    success = test_device_handling()
    sys.exit(0 if success else 1)
