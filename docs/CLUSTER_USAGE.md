# Running Experiments on Clusters

This guide explains how to run GTN experiments on SLURM (Titans) and HPC (DTU LSF) clusters using the Hydra configuration system.

## Quick Start

```bash
# Local run (default: MPO2, iris, NTN trainer)
python run.py

# Override model/dataset
python run.py model=lmpo2 dataset=abalone

# GTN trainer with custom learning rate
python run.py trainer=gtn trainer.lr=0.01

# Multirun sweep
python run.py --multirun model.bond_dim=4,6,8 seed=42,123,256
```

## Output Directory Structure

Hydra manages output directories automatically:

```
outputs/
├── 2026-03-15/                          # Single runs (by date)
│   ├── 14-30-00_MPO2_iris/              # {time}_{model}_{dataset}
│   │   ├── .hydra/
│   │   │   ├── config.yaml              # Resolved config
│   │   │   ├── hydra.yaml               # Hydra settings
│   │   │   └── overrides.yaml           # CLI overrides used
│   │   ├── run.log                      # Hydra log
│   │   └── results.json                 # Experiment results
│   └── ...
│
└── 2026-03-15_sweep/                    # Multirun sweeps
    ├── 0_MPO2_iris_seed42/
    ├── 1_MPO2_iris_seed123/
    └── ...
```

## Configuration Hierarchy

```
conf/
├── config.yaml          # Main config (defaults + hydra output settings)
├── model/               # Model configs (mpo2, lmpo2, mmpo2, + TypeI variants)
├── dataset/             # Dataset configs (21 UCI datasets)
│   └── size/            # Size presets with batch_size + cluster settings
│       ├── small.yaml   # batch_size=100, 3h SLURM / 6h HPC
│       ├── medium.yaml  # batch_size=64, 12h SLURM / 12h HPC
│       └── large.yaml   # batch_size=32, 24h SLURM / 24h HPC
├── trainer/             # Trainer configs (ntn, gtn)
└── experiment/          # Experiment presets for sweeps
```

## Dataset Size Presets

Each dataset imports a size preset that bundles:
- `batch_size` - Appropriate for dataset size
- `slurm.*` - SLURM settings (partition, time, mem, gpu_arch)
- `hpc.*` - HPC/LSF settings (queue, time, mem)

| Size   | Datasets | batch_size | SLURM time | HPC time |
|--------|----------|------------|------------|----------|
| small  | iris, wine, breast, hearth | 100 | 3:00:00 | 6:00 |
| medium | concrete, bike, obesity, ... | 64 | 12:00:00 | 12:00 |
| large  | abalone, adult, bank, ... | 32 | 24:00:00 | 24:00 |

## Experiment Configs

Experiment configs define sweep parameters and are located in `conf/experiment/`:

| Experiment | Description | Usage |
|------------|-------------|-------|
| `sanity_check` | Quick validation (6 models × 2 datasets, 1 epoch) | `+experiment=sanity_check` |
| `uci_ntn_sweep` | Full NTN sweep (L, bond_dim, seeds) | `+experiment=uci_ntn_sweep` |
| `uci_gtn_sweep` | Full GTN sweep (L, bond_dim, seeds) | `+experiment=uci_gtn_sweep` |

```bash
# Run sanity check (12 runs: 6 models × iris + concrete)
python run.py +experiment=sanity_check --multirun

# Run full NTN sweep on a specific dataset
python run.py +experiment=uci_ntn_sweep model=mpo2 dataset=abalone --multirun

# Run GTN sweep
python run.py +experiment=uci_gtn_sweep model=lmpo2 dataset=iris --multirun
```

## Cluster Submission

Use `scripts/generate_hydra_jobs.py` to generate job scripts for both clusters:

```bash
# Generate all jobs (21 datasets × 6 models = 126 jobs per cluster)
python scripts/generate_hydra_jobs.py

# SLURM only (Titans)
python scripts/generate_hydra_jobs.py --cluster slurm

# HPC only (DTU LSF)
python scripts/generate_hydra_jobs.py --cluster hpc

# Specific datasets/models
python scripts/generate_hydra_jobs.py --datasets iris,concrete --models mpo2,lmpo2

# GTN trainer
python scripts/generate_hydra_jobs.py --trainer gtn

# With experiment config
python scripts/generate_hydra_jobs.py --experiment uci_ntn_sweep
```

**Generated structure:**
```
jobs/
├── slurm/
│   ├── job_ntn_mpo2_iris.sh
│   ├── job_ntn_lmpo2_iris.sh
│   ├── ...
│   ├── logs/
│   └── submit_all.sh
└── hpc/
    ├── job_ntn_mpo2_iris.sh
    ├── ...
    ├── logs/
    └── submit_all.sh
```

**Submit jobs:**
```bash
# SLURM (Titans)
cd jobs/slurm && ./submit_all.sh
# Or individual: sbatch job_ntn_mpo2_iris.sh

# HPC (DTU LSF)
cd jobs/hpc && ./submit_all.sh
# Or individual: bsub < job_ntn_mpo2_iris.sh
```

Settings (time, mem, GPU) are pulled from dataset size presets in `conf/dataset/size/*.yaml`.

## Manual Job Scripts (Advanced)

Create a job script that calls `run.py` with Hydra overrides:

**SLURM (Titans):**
```bash
#!/bin/bash
#SBATCH --job-name=mpo2-iris
#SBATCH --output=logs/%x_%J.out
#SBATCH --error=logs/%x_%J.err
#SBATCH --partition=titans
#SBATCH --time=3:00:00
#SBATCH --mem=8gb
#SBATCH --gres=gpu:Ampere:1
#SBATCH --cpus-per-task=2

cd $HOME/GTN
source .venv/bin/activate

python run.py model=mpo2 dataset=iris trainer=ntn seed=42
```

**HPC (DTU LSF):**
```bash
#!/bin/sh
#BSUB -q gpuv100
#BSUB -J mpo2-iris
#BSUB -W 6:00
#BSUB -n 4
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -R "rusage[mem=500MB]"
#BSUB -o logs/%J.out
#BSUB -e logs/%J.err

cd $HOME/GTN
source .venv/bin/activate

python run.py model=mpo2 dataset=iris trainer=ntn seed=42
```

### Method 2: Multirun Sweep on Cluster

For parameter sweeps, use `--multirun` and submit the parent job:

```bash
#!/bin/bash
#SBATCH --job-name=sweep-iris
#SBATCH --time=12:00:00
#SBATCH --mem=16gb
#SBATCH --gres=gpu:Ampere:1

cd $HOME/GTN
source .venv/bin/activate

# Sweep runs sequentially on same node
python run.py --multirun \
    model=mpo2,lmpo2 \
    model.bond_dim=4,6,8 \
    seed=42,123,256
```

### Method 3: Array Jobs (Parallel Sweeps)

For parallel execution, use SLURM array jobs:

```bash
#!/bin/bash
#SBATCH --job-name=array-sweep
#SBATCH --array=0-29  # 30 jobs (6 models × 5 seeds)
#SBATCH --time=3:00:00
#SBATCH --mem=8gb
#SBATCH --gres=gpu:Ampere:1

cd $HOME/GTN
source .venv/bin/activate

# Map array index to parameters
MODELS=(mpo2 lmpo2 mmpo2 mpo2_typei lmpo2_typei mmpo2_typei)
SEEDS=(42 123 256 7 999)

MODEL_IDX=$((SLURM_ARRAY_TASK_ID / 5))
SEED_IDX=$((SLURM_ARRAY_TASK_ID % 5))

python run.py \
    model=${MODELS[$MODEL_IDX]} \
    dataset=iris \
    seed=${SEEDS[$SEED_IDX]}
```

## Accessing Cluster Settings in Config

The cluster settings are available in the config under `slurm.*` and `hpc.*`:

```python
# In run.py or custom scripts
@hydra.main(config_path="conf", config_name="config")
def main(cfg):
    # Access cluster settings (for logging/documentation)
    print(f"SLURM time: {cfg.slurm.time}")
    print(f"HPC queue: {cfg.hpc.queue}")
```

You can also print them via CLI:
```bash
python run.py --cfg job  # Show resolved config
python run.py --info defaults  # Show default composition
```

## CLI Override Examples

```bash
# Change model parameters
python run.py model.L=4 model.bond_dim=12

# Change trainer parameters
python run.py trainer.n_epochs=50 trainer.jitter_start=0.01

# Use experiment preset
python run.py +experiment=uci_ntn_sweep

# Override output directory
python run.py hydra.run.dir=outputs/custom_run

# Dry run (print config only)
python run.py --cfg job
```

## Completion Detection

The runner automatically skips completed runs:

- Checks for existing `results.json` in output directory
- Skips if `success: true` (logs previous val_quality)
- Skips permanently if `singular: true` (unrecoverable failure)
- Re-runs if previous attempt failed (non-singular)

**Override to force re-run:**
```bash
python run.py skip_completed=false ...
```

## Logs and Results

| File | Location | Content |
|------|----------|---------|
| Hydra log | `outputs/{date}/{time}_{exp}/run.log` | Python logging output |
| Config | `outputs/{date}/{time}_{exp}/.hydra/config.yaml` | Resolved configuration |
| Results | `outputs/{date}/{time}_{exp}/results.json` | Metrics, losses, config |
| SLURM logs | `logs/{job_name}_%J.out` | Cluster stdout/stderr |

## Migration from Old System

The old system used JSON configs in `experiments/configs/` and separate runner scripts. The new Hydra system:

| Old | New |
|-----|-----|
| `experiments/configs/uci_ntn_iris.json` | `python run.py dataset=iris trainer=ntn` |
| `experiments/run_grid_search.py` | `python run.py` |
| `experiments/run_grid_search_gtn.py` | `python run.py trainer=gtn` |
| Separate SLURM/HPC config files | Dataset size presets include cluster settings |
| Manual grid expansion | `--multirun` flag |

## Troubleshooting

**Config not found:**
```bash
# Check available configs
ls conf/model/    # Available models
ls conf/dataset/  # Available datasets
```

**Override syntax errors:**
```bash
# Wrong: spaces around =
python run.py model = lmpo2  # ✗

# Correct: no spaces
python run.py model=lmpo2    # ✓
```

**Multirun not working:**
```bash
# Wrong: missing --multirun flag
python run.py seed=1,2,3  # Treats as string "1,2,3"

# Correct: use --multirun
python run.py --multirun seed=1,2,3  # Runs 3 experiments
```
