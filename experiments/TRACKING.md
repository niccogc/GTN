# Experiment Tracking System

The training scripts support flexible experiment tracking through a callback-based system. Tracking is completely separated from training logic.

## Architecture

```
Training Script (train_*.py)
    ↓
NTN.fit() with callbacks
    ↓ (exposes metrics via callbacks)
Tracker (trackers.py)
    ↓ (saves to backend)
Backend (File / AIM / Both)
```

## Tracking Backends

### 1. File Tracking (default)

Saves epoch-by-epoch metrics to JSON files.

```bash
# Use file tracking (default)
TRACKER_BACKEND=file bash experiments/run_mpo2.sh

# Custom output directory
TRACKER_DIR=my_logs bash experiments/run_mpo2.sh
```

**Output structure:**
```json
{
  "experiment_name": "mpo2_experiment_seed0",
  "config": {...},
  "hparams": {...},
  "metrics_log": [
    {"step": -1, "loss": 6.64, "r2": -3.88},  // Init
    {"step": 0, "loss": 0.27, "r2": 0.80, "reg_loss": 0.86, "jitter": 0.05},  // Epoch 0
    {"step": 1, "loss": 0.26, "r2": 0.81, "reg_loss": 0.80, "jitter": 0.045}, // Epoch 1
    ...
  ],
  "summary": {
    "seed": 0,
    "loss": 0.271,
    "r2": 0.801,
    "success": true
  }
}
```

### 2. AIM Tracking

Track experiments to AIM server (local or remote).

```bash
# Local AIM (creates .aim directory)
TRACKER_BACKEND=aim bash experiments/run_mpo2.sh

# Remote AIM server
TRACKER_BACKEND=aim AIM_REPO="aim://192.168.5.5:5800" bash experiments/run_mpo2.sh

# With authentication (not on VPN)
TRACKER_BACKEND=aim \
  AIM_REPO="aim://aimtracking.kosmon.org:443" \
  CF_ACCESS_CLIENT_ID="xxx" \
  CF_ACCESS_CLIENT_SECRET="yyy" \
  bash experiments/run_mpo2.sh
```

**AIM Features:**
- Web UI for visualization
- Compare across experiments
- Search and filter runs
- Export results

### 3. Both (File + AIM)

Track to both file and AIM simultaneously.

```bash
TRACKER_BACKEND=both \
  TRACKER_DIR=logs \
  AIM_REPO="aim://192.168.5.5:5800" \
  bash experiments/run_mpo2.sh
```

### 4. None (No Tracking)

Disable tracking for quick tests.

```bash
TRACKER_BACKEND=none bash experiments/run_mpo2.sh
```

## Environment Variables

### Tracking Configuration

- `TRACKER_BACKEND` - Tracking backend: `file`, `aim`, `both`, `none` (default: `file`)
- `TRACKER_DIR` - Directory for file tracking (default: `experiment_logs`)
- `AIM_REPO` - AIM repository (default: `.aim` for local)
- `EXPERIMENT_NAME` - Base name for experiment (default: `{model}_experiment`)

### AIM Authentication (when not on VPN)

- `CF_ACCESS_CLIENT_ID` - Cloudflare Access Client ID
- `CF_ACCESS_CLIENT_SECRET` - Cloudflare Access Client Secret
- `AIM_AUTH_TOKEN` - Alternative: Bearer token
- `AIM_CUSTOM_HEADERS` - Alternative: Custom headers (JSON string)

## Metrics Tracked

### Per Epoch
- `loss` - Data loss (MSE or cross-entropy)
- `r2` - R² score (regression only)
- `reg_loss` - Regularized loss (loss + jitter × ||weights||²)
- `jitter` - Current jitter value
- `weight_norm_sq` - Squared Frobenius norm of weights

### Summary (Final)
- `seed` - Random seed used
- `loss` - Final loss
- `r2` - Final R² score
- `success` - Whether training succeeded (R² > 0)
- `error` - Error message if failed

## Callbacks in NTN

The `NTN.fit()` method supports two callbacks:

```python
def callback_init(scores, info):
    """Called after initial evaluation.
    
    Args:
        scores: dict of initial metrics
        info: dict with n_epochs, jitter_schedule, regularize
    """
    pass

def callback_epoch(epoch, scores, info):
    """Called after each epoch.
    
    Args:
        epoch: int (0-indexed)
        scores: dict of metrics from evaluate()
        info: dict with epoch, jitter, reg_loss, weight_norm_sq, best_reg_loss, is_best
    """
    pass

# Usage
ntn.fit(
    n_epochs=10,
    callback_init=callback_init,
    callback_epoch=callback_epoch
)
```

## Custom Trackers

You can implement custom trackers by extending `BaseTracker`:

```python
from experiments.trackers import BaseTracker

class MyCustomTracker(BaseTracker):
    def log_hparams(self, hparams):
        # Log hyperparameters
        pass
    
    def log_metrics(self, metrics, step):
        # Log metrics for a step
        pass
    
    def log_summary(self, summary):
        # Log final summary
        pass
    
    def finalize(self):
        # Cleanup
        pass
```

## Examples

### Basic File Tracking
```bash
bash experiments/run_mpo2.sh
# Creates: experiment_logs/mpo2_experiment_seed{0-4}.json
```

### Custom Experiment Name
```bash
EXPERIMENT_NAME="california_housing_L3_bond6" bash experiments/run_mpo2.sh
# Creates: experiment_logs/california_housing_L3_bond6_seed{0-4}.json
```

### Parameter Sweep with AIM
```bash
for bond_dim in 4 6 8 10; do
  TRACKER_BACKEND=aim \
  EXPERIMENT_NAME="bond_sweep_${bond_dim}" \
  BOND_DIM=$bond_dim \
  bash experiments/run_mpo2.sh
done

# View in AIM UI
aim up
```

### Multi-Backend Tracking
```bash
# Track locally AND to remote server
TRACKER_BACKEND=both \
  TRACKER_DIR=local_logs \
  AIM_REPO="aim://192.168.5.5:5800" \
  EXPERIMENT_NAME="production_run" \
  bash experiments/run_lmpo2.sh
```

## Viewing Results

### File Tracking
```bash
# View latest experiment
cat experiment_logs/mpo2_experiment_seed0.json | python -m json.tool

# Extract R² scores from all seeds
jq '.summary.r2' experiment_logs/mpo2_experiment_seed*.json

# Plot learning curve (requires pandas, matplotlib)
python -c "
import json
import pandas as pd
import matplotlib.pyplot as plt

with open('experiment_logs/mpo2_experiment_seed0.json') as f:
    data = json.load(f)

df = pd.DataFrame(data['metrics_log'])
plt.plot(df['step'], df['loss'], label='loss')
plt.plot(df['step'], df['r2'], label='r2')
plt.legend()
plt.show()
"
```

### AIM Tracking
```bash
# Start AIM UI
aim up

# Open browser to http://localhost:43800
```

## Notes

- Each seed gets its own experiment/tracker instance
- File tracking creates one JSON per seed
- AIM tracking creates one run per seed (grouped by experiment name)
- Tracking has zero impact on training performance
- Callbacks are called synchronously after evaluation
- Failed runs are still tracked (with error info in summary)
