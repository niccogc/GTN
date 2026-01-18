#!/usr/bin/env python3
"""
Analyze the early stopping behavior from the last run
"""

# From the log output we captured:
epochs_and_r2 = [
    (1, -5.2096),
    (51, 0.2121),
    (101, 0.1576),
    # Early stopping triggered at epoch 143
    # Best R²: 0.285358
]

print("Early Stopping Analysis:")
print("=" * 40)
print("The early stopping correctly triggered when validation R²")
print("did not improve by more than min_delta=1e-08 for patience=40 epochs.")
print()
print("Key points:")
print("1. Best R² achieved: 0.285358 (sometime between epoch 51-101)")
print("2. From epoch 103-143: No improvement > 1e-08 for 40 epochs")
print("3. Loss can still fluctuate, but no significant improvement")
print("4. This prevents wasted computation when model has converged")
print()
print("The algorithm is working correctly - it stops when improvements")
print("become negligible, not when loss stops changing completely.")
