# type: ignore
"""
Tensor Network Training Framework

This package provides implementations of tensor network models for machine learning
using both Newton-based (NTN) and Gradient-based (GTN) training methods.

Structure:
- base: Core training classes (GTN, NTN)
- standard: Standard tensor network architectures (MPO2, LMPO2, MMPO2, MPS)
- typeI: Type I ensemble models with varying number of sites
- builder: Data loading utilities
- losses: Loss functions with derivatives
- utils: Helper functions and metrics
"""

from model.base import GTN, NTN
from model.standard import MPO2, LMPO2, MMPO2
from model.typeI import (
    MPO2TypeI,
    LMPO2TypeI,
    MMPO2TypeI,
    MPO2TypeI_GTN,
    LMPO2TypeI_GTN,
    MMPO2TypeI_GTN,
)

__all__ = [
    "GTN",
    "NTN",
    "MPO2",
    "LMPO2",
    "MMPO2",
    "MPO2TypeI",
    "LMPO2TypeI",
    "MMPO2TypeI",
    "MPO2TypeI_GTN",
    "LMPO2TypeI_GTN",
    "MMPO2TypeI_GTN",
]
