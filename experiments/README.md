# Production Training Scripts for MPO2 Models

This directory contains production-ready training scripts for MPO2, LMPO2, and MMPO2 models with multi-seed analysis and environment variable configuration.

## Quick Start

```bash
# Train MPO2 with default settings
bash experiments/run_mpo2.sh

# Train LMPO2 with custom settings
SEEDS="0,1,2" N_EPOCHS=20 bash experiments/run_lmpo2.sh

# Train MMPO2 with verbose output
VERBOSE=true bash experiments/run_mmpo2.sh
```

## Available Scripts

- `run_mpo2.sh` - Train simple MPS with output dimension
- `run_lmpo2.sh` - Train linear MPO2 with dimensionality reduction  
- `run_mmpo2.sh` - Train masked MPO2 with cumulative mask

## Environment Variables

### Dataset
- `DATASET` - Dataset name (default: `california_housing`)
- `N_SAMPLES` - Number of samples to use (default: `500`)

### Model Architecture
- `L` - Number of sites (default: `3`)
- `BOND_DIM` - Bond dimension (default: `6`)
- `OUTPUT_SITE` - Which site has output (default: `1`)
- `INIT_STRENGTH` - Initialization strength (default: `0.1`)
- `REDUCTION_FACTOR` - *LMPO2 only* - Reduction factor (default: `0.5`)

### Training
- `BATCH_SIZE` - Batch size (default: `100`)
- `N_EPOCHS` - Number of epochs (default: `10`)
- `JITTER_START` - Initial jitter value (default: `0.05`)
- `JITTER_DECAY` - Jitter decay rate (default: `0.9`)
- `JITTER_MIN` - Minimum jitter (default: `1e-6`)

### Multi-Seed
- `SEEDS` - Comma-separated list of seeds (default: `0,1,2,3,4`)

### Output
- `OUTPUT_FILE` - JSON output file (default: `results_{model}.json`)
- `VERBOSE` - Print training progress (default: `false`)

## Output Format

Results are saved as JSON with the following structure:

```json
{
  "model": "MPO2",
  "n_total": 5,
  "n_success": 5,
  "loss_mean": 0.2625,
  "loss_std": 0.0062,
  "r2_mean": 0.8072,
  "r2_std": 0.0046,
  "results": [
    {"seed": 0, "loss": 0.271, "r2": 0.801, "success": true},
    ...
  ],
  "config": {...},
  "dataset_info": {...}
}
```

## Statistics

- Only runs with positive R² (at least one successful sweep) are included in statistics
- Mean and standard deviation computed over successful runs only
- Failed runs are tracked but excluded from final statistics

## Example: Parameter Sweep

```bash
# Test different bond dimensions
for bond_dim in 4 6 8; do
  BOND_DIM=$bond_dim OUTPUT_FILE="results_mpo2_bond${bond_dim}.json" bash experiments/run_mpo2.sh
done

# Test different jitter schedules
for jitter in 0.01 0.05 0.1; do
  JITTER_START=$jitter OUTPUT_FILE="results_mpo2_jitter${jitter}.json" bash experiments/run_mpo2.sh
done
```

## Adding New Datasets

Edit `experiments/dataset_loader.py` and add your dataset loading logic:

```python
def load_dataset(dataset_name, n_samples=None, seed=0):
    if dataset_name == "my_dataset":
        # Load your data here
        X = ...  # shape: (n_samples, n_features)
        y = ...  # shape: (n_samples, 1)
        
        # Standardize, convert to torch, return
        ...
        return X, y, dataset_info
```

Then use it:
```bash
DATASET="my_dataset" bash experiments/run_mpo2.sh
```

## File Structure

```
experiments/
├── README.md              # This file
├── dataset_loader.py      # Dataset loading utilities
├── train_mpo2.py          # MPO2 training script
├── train_lmpo2.py         # LMPO2 training script  
├── train_mmpo2.py         # MMPO2 training script
├── run_mpo2.sh            # MPO2 bash wrapper
├── run_lmpo2.sh           # LMPO2 bash wrapper
└── run_mmpo2.sh           # MMPO2 bash wrapper
```

## Notes

- All scripts use regularized loss (data_loss + jitter × ||weights||²) for best model selection
- Best model is automatically restored at end of training
- Jitter schedule: `jitter_start × jitter_decay^epoch` (clipped to `jitter_min`)
- Input features automatically get bias term appended via `create_inputs()`
