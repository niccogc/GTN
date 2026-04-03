#!/usr/bin/env python3
"""Generate Hydra dataset configs from UCI dataset registry."""

from pathlib import Path

CONF_DIR = Path(__file__).parent.parent / "conf" / "dataset"

# UCI datasets: (name, task)
UCI_DATASETS = [
    ("student_perf", "regression"),
    ("abalone", "regression"),
    ("obesity", "regression"),
    ("bike", "regression"),
    ("realstate", "regression"),
    ("energy_efficiency", "regression"),
    ("concrete", "regression"),
    ("ai4i", "regression"),
    ("appliances", "regression"),
    ("popularity", "regression"),
    ("iris", "classification"),
    ("hearth", "classification"),
    ("winequalityc", "classification"),
    ("breast", "classification"),
    ("adult", "classification"),
    ("bank", "classification"),
    ("wine", "classification"),
    ("car_evaluation", "classification"),
    ("student_dropout", "classification"),
    ("mushrooms", "classification"),
    ("seoulBike", "regression"),
]

# Dataset size classifications
DATASET_SIZES = {
    "iris": "small",
    "wine": "small",
    "breast": "small",
    "hearth": "small",
    "car_evaluation": "medium",
    "student_perf": "medium",
    "realstate": "medium",
    "concrete": "medium",
    "energy_efficiency": "medium",
    "winequalityc": "medium",
    "obesity": "medium",
    "bike": "medium",
    "seoulBike": "medium",
    "abalone": "large",
    "ai4i": "large",
    "appliances": "large",
    "popularity": "large",
    "adult": "large",
    "bank": "large",
    "student_dropout": "large",
    "mushrooms": "large",
}

BASE_TEMPLATE = """# @package _global_
# Base dataset configuration

dataset:
  name: ???
  task: ???
  cap: 50
"""

DATASET_TEMPLATE = """# @package _global_
defaults:
  - _base
  - size/{size}

dataset:
  name: {name}
  task: {task}
"""


def main():
    CONF_DIR.mkdir(parents=True, exist_ok=True)

    (CONF_DIR / "_base.yaml").write_text(BASE_TEMPLATE)
    print(f"Created: _base.yaml")

    for name, task in UCI_DATASETS:
        size = DATASET_SIZES.get(name, "medium")
        (CONF_DIR / f"{name}.yaml").write_text(
            DATASET_TEMPLATE.format(name=name, task=task, size=size)
        )
        print(f"Created: {name}.yaml (size={size})")

    print(f"\nGenerated {len(UCI_DATASETS) + 1} configs")


if __name__ == "__main__":
    main()
