# Experiment Configuration Files

This directory contains JSON configuration files for running parameter grid searches across different datasets and models.

## Configuration Structure

```json
{
  "experiment_name": "unique_experiment_name",
  "dataset": "dataset_name",
  "task": "regression or classification",
  
  "parameter_grid": {
    "param1": [value1, value2, ...],
    "param2": [value1, value2, ...],
    ...
  },
  
  "fixed_params": {
    "param1": value,
    "param2": value,
    ...
  },
  
  "tracker": {
    "backend": "aim",
    "tracker_dir": "experiment_logs",
    "aim_repo": null
  },
  
  "output": {
    "results_dir": "results/experiment_name",
    "save_models": false,
    "save_individual_runs": true
  }
}
```

## Parameter Grid

The `parameter_grid` section defines parameters to sweep over. Each parameter can have a list of values. The experiment runner will create all combinations (Cartesian product) of these parameters.

### Common Grid Parameters

**Model Architecture:**
- `model`: Model type - `["MPO2", "LMPO2", "MMPO2"]`
- `L`: Number of sites - `[2, 3, 4, 5, ...]`
- `bond_dim`: Bond dimension - `[4, 6, 8, 10, ...]`
- `output_site`: Which site has output - `[0, 1, 2, ...]`
- `init_strength`: Initialization strength - `[0.1, 0.01, 0.001, ...]`

**Model-Specific:**
- `rank`: Rank for LMPO2/MMPO2 - `[3, 5, 7, ...]`
- `reduction_factor`: Reduction factor for LMPO2 - `[0.3, 0.5, 0.7]`

**Training:**
- `jitter_start`: Initial jitter/regularization - `[1e-1, 1e-2, 1e-5, ...]`
- `jitter_decay`: Jitter decay rate - `[0.9, 0.95, 0.99]`
- `max_sweeps`: Maximum number of sweeps - `[3, 10, 20, 50]`
- `batch_size`: Batch size - `[50, 100, 200]`
- `n_epochs`: Number of epochs - `[10, 20, 50]`

## Fixed Parameters

The `fixed_params` section defines parameters that remain constant across all experiments.

### Common Fixed Parameters

- `seeds`: List of random seeds for multi-seed analysis - `[0, 1, 2, 3, 4]`
- `verbose`: Print training progress - `true` or `false`
- `adaptive_jitter`: Use adaptive jitter - `true` or `false`
- `patience`: Early stopping patience (epochs) - `5` or `null` for no early stopping
- `min_delta`: Minimum improvement for early stopping - `0.001`
- `train_selection`: Use train selection strategy - `true` or `false`

## Tracker Configuration

Specify how to track experiments:

```json
"tracker": {
  "backend": "aim",           // "aim", "wandb", "mlflow", "file", "both", "none"
  "tracker_dir": "logs",      // Directory for file-based tracking
  "aim_repo": null            // AIM repository path (null = local .aim)
}
```

## Output Configuration

Control where results are saved:

```json
"output": {
  "results_dir": "results/experiment_name",  // Where to save results
  "save_models": false,                      // Save trained model weights
  "save_individual_runs": true               // Save JSON for each run
}
```

## Example Configurations

### Regression Grid Search
```json
{
  "experiment_name": "abalone_grid",
  "dataset": "abalone",
  "task": "regression",
  "parameter_grid": {
    "model": ["MPO2", "LMPO2"],
    "L": [2, 3, 4],
    "bond_dim": [4, 6, 8],
    "jitter_start": [1e-2, 1e-3, 1e-4]
  },
  "fixed_params": {
    "seeds": [0, 1, 2, 3, 4],
    "n_epochs": 50,
    "batch_size": 100
  }
}
```

### Classification Grid Search
```json
{
  "experiment_name": "iris_classification",
  "dataset": "iris",
  "task": "classification",
  "parameter_grid": {
    "model": ["MPO2"],
    "L": [2, 3],
    "jitter_start": [5.0, 1.0, 0.5]
  },
  "fixed_params": {
    "seeds": [0, 1, 2],
    "n_epochs": 30,
    "jitter_decay": 0.1,
    "train_selection": true
  }
}
```

## Running Experiments

```bash
# Run a single configuration file
python experiments/run_grid_search.py --config experiments/configs/abalone_grid.json

# Resume a partially completed experiment
python experiments/run_grid_search.py --config experiments/configs/abalone_grid.json --resume

# Dry run (show what would be executed without running)
python experiments/run_grid_search.py --config experiments/configs/abalone_grid.json --dry-run
```

## Notes

- **Grid Size**: Number of experiments = product of all parameter grid lengths × number of seeds
  - Example: 3 models × 4 L values × 3 bond_dims × 5 seeds = 180 experiments
- **Time Estimate**: Each experiment typically takes 1-5 minutes depending on dataset size and n_epochs
- **Resume**: The runner automatically skips completed experiments if results exist
- **Tracking**: All experiments within the same config file share one experiment name but have unique run names
