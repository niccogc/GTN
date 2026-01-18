# type: ignore
"""
Configuration parser for grid search experiments.
Handles parameter grid expansion and validation.
"""

import json
import itertools
from typing import Dict, List, Any, Tuple


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from JSON file."""
    with open(config_path, "r") as f:
        config = json.load(f)

    # Validate required fields
    required_fields = ["experiment_name", "dataset", "parameter_grid"]
    missing = [f for f in required_fields if f not in config]
    if missing:
        raise ValueError(f"Missing required fields in config: {missing}")

    # Set defaults
    if "fixed_params" not in config:
        config["fixed_params"] = {}

    if "tracker" not in config:
        config["tracker"] = {"backend": "file", "tracker_dir": "experiment_logs", "aim_repo": None}

    if "output" not in config:
        config["output"] = {
            "results_dir": f"results/{config['experiment_name']}",
            "save_models": False,
            "save_individual_runs": True,
        }

    return config


def expand_parameter_grid(parameter_grid: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
    """
    Expand parameter grid into list of all combinations.

    Args:
        parameter_grid: Dict mapping parameter names to lists of values

    Returns:
        List of dicts, each representing one parameter combination

    Example:
        >>> grid = {'L': [2, 3], 'bond_dim': [4, 6]}
        >>> expand_parameter_grid(grid)
        [
            {'L': 2, 'bond_dim': 4},
            {'L': 2, 'bond_dim': 6},
            {'L': 3, 'bond_dim': 4},
            {'L': 3, 'bond_dim': 6}
        ]
    """
    # Extract parameter names and their value lists
    param_names = list(parameter_grid.keys())
    param_values = [parameter_grid[name] for name in param_names]

    # Generate Cartesian product
    combinations = list(itertools.product(*param_values))

    # Convert to list of dicts
    expanded = [dict(zip(param_names, combo)) for combo in combinations]

    return expanded


def merge_params(grid_params: Dict[str, Any], fixed_params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge grid parameters with fixed parameters.
    Grid parameters take precedence over fixed parameters.
    """
    merged = fixed_params.copy()
    merged.update(grid_params)
    return merged


def generate_run_name(grid_params: Dict[str, Any], seed: int) -> str:
    """
    Generate a unique run name from grid parameters and seed.

    Format: model_L{L}_d{bond_dim}_j{jitter}_s{seed}
    Example: MPO2_L3_d6_j0.01_s0
    """
    model = grid_params.get("model", "MPO2")
    L = grid_params.get("L", "X")
    bond_dim = grid_params.get("bond_dim", "X")
    jitter = grid_params.get("jitter_start", "X")

    # Format jitter in scientific notation if small
    if isinstance(jitter, (int, float)):
        if jitter < 0.01:
            jitter_str = f"{jitter:.0e}".replace("-", "m")  # 1e-5 -> 1em5
        else:
            jitter_str = f"{jitter:.3f}".rstrip("0").rstrip(".")
    else:
        jitter_str = str(jitter)

    return f"{model}_L{L}_d{bond_dim}_j{jitter_str}_s{seed}"


def generate_run_id(grid_params: Dict[str, Any], seed: int) -> str:
    """
    Generate a unique, filesystem-safe run ID from grid parameters and seed.

    Format: model-L{L}-d{bond_dim}-init{init}-jit{jitter}-sweep{sweeps}-rank{rank}-seed{seed}
    Example: MPO2-L3-d6-init0.1-jit0.01-sweep10-rank5-seed0

    This is used for:
    - Result file naming
    - Checking if a run has already been completed
    """
    parts = []

    # Always include model
    parts.append(grid_params.get("model", "MPO2"))

    # Core architecture parameters
    if "L" in grid_params:
        parts.append(f"L{grid_params['L']}")
    if "bond_dim" in grid_params:
        parts.append(f"d{grid_params['bond_dim']}")
    if "output_site" in grid_params:
        parts.append(f"out{grid_params['output_site']}")

    # Initialization
    if "init_strength" in grid_params:
        init_val = grid_params["init_strength"]
        if init_val < 0.01:
            parts.append(f"init{init_val:.0e}".replace("-", "m"))
        else:
            parts.append(f"init{init_val:.3f}".rstrip("0").rstrip("."))

    # Regularization/jitter
    if "jitter_start" in grid_params:
        jitter_val = grid_params["jitter_start"]
        if jitter_val < 0.01:
            parts.append(f"jit{jitter_val:.0e}".replace("-", "m"))
        else:
            parts.append(f"jit{jitter_val:.3f}".rstrip("0").rstrip("."))

    # Training parameters
    if "max_sweeps" in grid_params:
        parts.append(f"sweep{grid_params['max_sweeps']}")

    # Model-specific parameters
    if "rank" in grid_params:
        parts.append(f"rank{grid_params['rank']}")
    if "reduction_factor" in grid_params:
        rf_val = grid_params["reduction_factor"]
        parts.append(f"rf{rf_val:.2f}".rstrip("0").rstrip("."))

    # Always include seed last
    parts.append(f"seed{seed}")

    return "-".join(parts)


def validate_model_params(model: str, params: Dict[str, Any]) -> None:
    """
    Validate that model-specific parameters are present.

    Raises:
        ValueError: If required parameters for model are missing
    """
    # Common required parameters for all models
    common_required = ["L", "bond_dim", "output_site"]

    model_specific = {
        "MPO2": [],
        "LMPO2": ["rank", "reduction_factor"],
        "MMPO2": ["rank"],
        "MPO2TypeI": [],
        "LMPO2TypeI": ["rank", "reduction_factor"],
        "MMPO2TypeI": [],
        "MPO2TypeI_GTN": [],
        "LMPO2TypeI_GTN": ["rank", "reduction_factor"],
        "MMPO2TypeI_GTN": [],
    }

    if model not in model_specific:
        raise ValueError(f"Unknown model: {model}. Valid models: {list(model_specific.keys())}")

    # Check common required parameters
    missing = [p for p in common_required if p not in params]
    if missing:
        raise ValueError(f"Missing required parameters for {model}: {missing}")

    # Check model-specific parameters
    missing = [p for p in model_specific[model] if p not in params]
    if missing:
        raise ValueError(f"Missing required parameters for {model}: {missing}")


def get_default_params_for_model(model: str) -> Dict[str, Any]:
    """
    Get default parameters for a specific model.
    These are reasonable defaults if not specified in config.
    """
    defaults = {
        "MPO2": {"output_site": 1, "init_strength": 0.1},
        "LMPO2": {"output_site": 1, "init_strength": 0.1, "reduction_factor": 0.5, "rank": 5},
        "MMPO2": {"output_site": 1, "init_strength": 0.1, "rank": 5},
        "MPO2TypeI": {"output_site": 1, "init_strength": 0.1},
        "LMPO2TypeI": {"output_site": 1, "init_strength": 0.1, "reduction_factor": 0.5, "rank": 5},
        "MMPO2TypeI": {"output_site": 1, "init_strength": 0.1},
        "MPO2TypeI_GTN": {"output_site": 1, "init_strength": 0.1},
        "LMPO2TypeI_GTN": {
            "output_site": 1,
            "init_strength": 0.1,
            "reduction_factor": 0.5,
            "rank": 5,
        },
        "MMPO2TypeI_GTN": {"output_site": 1, "init_strength": 0.1},
    }

    return defaults.get(model, {})


def create_experiment_plan(config: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Create complete experiment plan from config.

    Returns:
        experiments: List of experiment configs (one per grid combination × seed)
        metadata: Dict with experiment metadata (total count, grid size, etc.)
    """
    grid_combinations = expand_parameter_grid(config["parameter_grid"])

    seeds = config["fixed_params"].get("seeds", [0])
    if not isinstance(seeds, list):
        seeds = [seeds]

    experiments = []

    for grid_params in grid_combinations:
        model = grid_params.get("model", "MPO2")
        defaults = get_default_params_for_model(model)

        full_params = {}
        full_params.update(defaults)
        full_params.update(config["fixed_params"])
        full_params.update(grid_params)

        for seed in seeds:
            try:
                validate_model_params(model, full_params)
            except ValueError as e:
                print(f"Warning: Skipping invalid configuration: {e}")
                continue

            experiment = {
                "experiment_name": config["experiment_name"],
                "dataset": config["dataset"],
                "task": config.get("task", "regression"),
                "params": full_params,
                "seed": seed,
                "run_name": generate_run_name(grid_params, seed),
                "run_id": generate_run_id(grid_params, seed),
                "grid_params": grid_params,
                "tracker": config["tracker"],
                "output": config["output"],
            }

            experiments.append(experiment)

    # Create metadata
    metadata = {
        "total_experiments": len(experiments),
        "grid_size": len(grid_combinations),
        "n_seeds": len(seeds),
        "seeds": seeds,
        "parameter_grid": config["parameter_grid"],
        "fixed_params": config["fixed_params"],
    }

    return experiments, metadata


def print_experiment_summary(experiments: List[Dict[str, Any]], metadata: Dict[str, Any]) -> None:
    """Print a summary of the experiment plan."""
    print("=" * 70)
    print("EXPERIMENT PLAN SUMMARY")
    print("=" * 70)
    print(f"Total experiments: {metadata['total_experiments']}")
    print(f"Grid combinations: {metadata['grid_size']}")
    print(f"Seeds per combination: {metadata['n_seeds']}")
    print()

    print("Parameter Grid:")
    for param, values in metadata["parameter_grid"].items():
        print(f"  {param}: {values} (n={len(values)})")
    print()

    print("Fixed Parameters:")
    for param, value in metadata["fixed_params"].items():
        if param != "seeds":
            print(f"  {param}: {value}")
    print()

    # Show first few experiment names
    print("Example runs (first 5):")
    for i, exp in enumerate(experiments[:5]):
        print(f"  {i + 1}. {exp['run_id']}")

    if len(experiments) > 5:
        print(f"  ... ({len(experiments) - 5} more)")
    print()
