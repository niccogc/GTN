# GTN Experiment Infrastructure Migration Plan

## Current State

After cleanup, the experiment infrastructure is organized as:

```
experiments/
├── run_grid_search.py          # NTN (Newton-based) runner
├── run_grid_search_gtn.py      # GTN (Gradient-based) runner
├── run_grid_search_profiled.py # Profiled NTN runner
├── configs/                    # 90 JSON config files
│   ├── uci_ntn_*.json
│   └── uci_gtn_*.json
├── utils/
│   ├── config_parser.py        # JSON loading + grid expansion
│   ├── dataset_loader.py       # UCI dataset loading
│   ├── device_utils.py         # CUDA/CPU handling
│   └── trackers.py             # AIM/file tracking
├── jobs/                       # HPC job generation
└── results/                    # Experiment outputs
```

### Current Pain Points

1. **Manual grid expansion** - `config_parser.py` implements custom Cartesian product logic
2. **No config composition** - Each JSON file is standalone, lots of duplication
3. **No CLI overrides** - Must edit JSON files to change parameters
4. **No automatic sweeps** - Grid search logic baked into runner scripts
5. **No experiment grouping** - Hard to organize related experiments
6. **Limited reproducibility** - Config not automatically saved with results

---

## Why Hydra?

### Benefits

| Feature | Current | With Hydra |
|---------|---------|------------|
| Config composition | ❌ Copy-paste | ✅ `defaults: [model/mpo2, dataset/uci]` |
| CLI overrides | ❌ Edit JSON | ✅ `python run.py model.bond_dim=8` |
| Grid search | ❌ Custom code | ✅ `--multirun model.L=2,3,4` |
| Sweeps | ❌ Manual | ✅ `hydra/sweeper=optuna` |
| Output organization | ❌ Manual | ✅ Auto `outputs/YYYY-MM-DD/HH-MM-SS/` |
| Config saving | ❌ Manual | ✅ Auto `.hydra/config.yaml` |
| Launcher plugins | ❌ Custom HPC scripts | ✅ `hydra/launcher=submitit_slurm` |

### Key Wins

1. **Single runner script** - One `run.py` instead of 3 separate scripts
2. **Composable configs** - Share common settings across NTN/GTN/datasets
3. **Native SLURM support** - Replace `generate_slurm_jobs.py` with `submitit` launcher
4. **Optuna integration** - Hyperparameter optimization out of the box
5. **Structured outputs** - Every run gets its own directory with full config

---

## Migration Plan

### Phase 1: Setup & Scaffolding

#### 1.1 Install Dependencies

```bash
# Add to pyproject.toml or requirements.txt
hydra-core>=1.3
hydra-submitit-launcher>=1.2  # For SLURM
hydra-optuna-sweeper>=1.2     # For HPO (optional)
omegaconf>=2.3
```

#### 1.2 Create Config Structure

```
conf/
├── config.yaml              # Main config with defaults
├── model/
│   ├── mpo2.yaml
│   ├── lmpo2.yaml
│   ├── mmpo2.yaml
│   ├── mpo2_typei.yaml
│   └── _gtn_base.yaml       # Shared GTN settings
├── dataset/
│   ├── uci_base.yaml        # Common UCI settings
│   ├── abalone.yaml
│   ├── concrete.yaml
│   └── ...
├── trainer/
│   ├── ntn.yaml             # Newton-based training
│   └── gtn.yaml             # Gradient-based training
├── tracker/
│   ├── aim.yaml
│   ├── file.yaml
│   └── none.yaml
├── hydra/
│   └── launcher/
│       ├── local.yaml
│       └── slurm_titans.yaml
└── experiment/              # Full experiment presets
    ├── ntn_uci_grid.yaml
    └── gtn_uci_grid.yaml
```

#### 1.3 Example Configs

**conf/config.yaml** (main entry point):
```yaml
defaults:
  - model: mpo2
  - dataset: abalone
  - trainer: ntn
  - tracker: file
  - _self_

seed: 42
output_dir: ${hydra:runtime.output_dir}

hydra:
  run:
    dir: outputs/${now:%Y-%m-%d}/${now:%H-%M-%S}
  sweep:
    dir: outputs/${now:%Y-%m-%d}_sweep
    subdir: ${hydra.job.num}
```

**conf/model/mpo2.yaml**:
```yaml
# @package _global_
model:
  name: MPO2
  L: 3
  bond_dim: 6
  init_strength: 0.1
  output_site: null  # defaults to L-1
```

**conf/model/lmpo2.yaml**:
```yaml
# @package _global_
defaults:
  - mpo2  # Inherit from MPO2

model:
  name: LMPO2
  reduction_factor: 0.5
  bond_dim_mpo: 2
```

**conf/trainer/ntn.yaml**:
```yaml
# @package _global_
trainer:
  type: ntn
  n_epochs: 10
  batch_size: 100
  jitter_start: 0.001
  jitter_decay: 0.95
  jitter_min: 0.001
  adaptive_jitter: true
  patience: 5
  min_delta: 0.001
```

**conf/trainer/gtn.yaml**:
```yaml
# @package _global_
trainer:
  type: gtn
  n_epochs: 50
  batch_size: 32
  lr: 0.001
  weight_decay: 0.01
  optimizer: adam
  patience: 10
```

**conf/dataset/abalone.yaml**:
```yaml
# @package _global_
defaults:
  - uci_base

dataset:
  name: abalone
  task: regression
```

### Phase 2: Unified Runner

#### 2.1 Create Single Entry Point

**run.py**:
```python
# type: ignore
"""Unified experiment runner with Hydra configuration."""

import hydra
from omegaconf import DictConfig, OmegaConf
import torch

from experiments.utils import load_dataset, create_tracker, DEVICE
from model.base.NTN import NTN
from model.base.GTN import GTN
from model.standard import MPO2, LMPO2, MMPO2
from model.typeI import MPO2TypeI, LMPO2TypeI, MMPO2TypeI

torch.set_default_dtype(torch.float64)


def create_model(cfg: DictConfig, input_dim: int, output_dim: int):
    """Create model from config."""
    model_map = {
        "MPO2": MPO2,
        "LMPO2": LMPO2,
        "MMPO2": MMPO2,
        "MPO2TypeI": MPO2TypeI,
        "LMPO2TypeI": LMPO2TypeI,
        "MMPO2TypeI": MMPO2TypeI,
    }
    
    model_cls = model_map[cfg.model.name]
    return model_cls(
        L=cfg.model.L,
        bond_dim=cfg.model.bond_dim,
        phys_dim=input_dim,
        output_dim=output_dim,
        output_site=cfg.model.get("output_site"),
        init_strength=cfg.model.get("init_strength", 0.1),
    )


def run_ntn(cfg: DictConfig, model, data, tracker):
    """Run Newton-based training."""
    # ... NTN training logic from run_grid_search.py
    pass


def run_gtn(cfg: DictConfig, model, data, tracker):
    """Run gradient-based training."""
    # ... GTN training logic from run_grid_search_gtn.py
    pass


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    
    # Set seed
    torch.manual_seed(cfg.seed)
    
    # Load data
    data, dataset_info = load_dataset(cfg.dataset.name)
    input_dim = data["X_train"].shape[1]
    output_dim = dataset_info.get("n_classes", 1)
    
    # Create model
    model = create_model(cfg, input_dim, output_dim)
    
    # Create tracker
    tracker = create_tracker(
        experiment_name=f"{cfg.model.name}_{cfg.dataset.name}",
        config=OmegaConf.to_container(cfg),
        backend=cfg.tracker.backend,
    )
    
    # Run training
    if cfg.trainer.type == "ntn":
        result = run_ntn(cfg, model, data, tracker)
    else:
        result = run_gtn(cfg, model, data, tracker)
    
    tracker.finalize()
    return result


if __name__ == "__main__":
    main()
```

#### 2.2 Usage Examples

```bash
# Single run with defaults
python run.py

# Override model
python run.py model=lmpo2 model.bond_dim=8

# Override dataset
python run.py dataset=concrete

# GTN instead of NTN
python run.py trainer=gtn

# Grid search (multirun)
python run.py --multirun \
    model=mpo2,lmpo2 \
    model.L=2,3,4 \
    model.bond_dim=4,6,8 \
    seed=0,1,2

# SLURM submission
python run.py --multirun \
    hydra/launcher=slurm_titans \
    model.L=2,3,4 \
    model.bond_dim=4,6,8
```

### Phase 3: SLURM Integration

#### 3.1 Submitit Launcher Config

**conf/hydra/launcher/slurm_titans.yaml**:
```yaml
# @package hydra.launcher
_target_: hydra_plugins.hydra_submitit_launcher.submitit_launcher.SlurmLauncher

submitit_folder: ${hydra.sweep.dir}/.submitit/%j

timeout_min: 1440  # 24 hours
cpus_per_task: 4
gpus_per_node: 1
tasks_per_node: 1
mem_gb: 32
nodes: 1
partition: gpu

# SLURM-specific
slurm_additional_parameters:
  mail-type: END,FAIL
  mail-user: ${oc.env:USER}@example.com
```

#### 3.2 Replace Job Generation Scripts

The `jobs/generate_hpc_jobs.py` and `jobs/generate_slurm_jobs.py` become obsolete - Hydra's submitit launcher handles this automatically.

### Phase 4: Testing & Validation

#### 4.1 Config Validation Tests

```python
# tests/test_configs.py
import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

@pytest.fixture
def hydra_config():
    with initialize_config_dir(config_dir="../conf"):
        yield

def test_all_model_configs_load(hydra_config):
    for model in ["mpo2", "lmpo2", "mmpo2", "mpo2_typei"]:
        cfg = compose(config_name="config", overrides=[f"model={model}"])
        assert cfg.model.name is not None
        assert cfg.model.L > 0

def test_all_dataset_configs_load(hydra_config):
    for dataset in ["abalone", "concrete", "iris", "wine"]:
        cfg = compose(config_name="config", overrides=[f"dataset={dataset}"])
        assert cfg.dataset.name == dataset

def test_grid_search_expansion(hydra_config):
    cfg = compose(
        config_name="config",
        overrides=["model.L=2,3", "model.bond_dim=4,6"]
    )
    # Verify multirun would produce 4 combinations
```

#### 4.2 Integration Tests

```python
# tests/test_runner.py
def test_ntn_single_run():
    """Verify NTN training runs without error."""
    # Run with minimal epochs
    result = subprocess.run([
        "python", "run.py",
        "trainer.n_epochs=1",
        "dataset=iris",
        "model.L=2",
        "model.bond_dim=4"
    ], capture_output=True)
    assert result.returncode == 0

def test_gtn_single_run():
    """Verify GTN training runs without error."""
    result = subprocess.run([
        "python", "run.py",
        "trainer=gtn",
        "trainer.n_epochs=1",
        "dataset=iris",
        "model.L=2",
        "model.bond_dim=4"
    ], capture_output=True)
    assert result.returncode == 0
```

#### 4.3 Migration Validation

```bash
# Compare old vs new results
# 1. Run old system
python experiments/run_grid_search.py --config experiments/configs/uci_ntn_iris.json --dry-run

# 2. Run new system
python run.py dataset=iris model=mpo2 --info

# 3. Verify parameter expansion matches
```

---

## Migration Checklist

### Phase 1: Setup
- [ ] Add Hydra dependencies to project
- [ ] Create `conf/` directory structure
- [ ] Convert 1 JSON config to Hydra YAML (e.g., `uci_ntn_iris.json`)
- [ ] Test config loading with `python -c "from hydra import compose, initialize..."`

### Phase 2: Runner
- [ ] Create `run.py` with Hydra decorator
- [ ] Extract common training logic from `run_grid_search.py` and `run_grid_search_gtn.py`
- [ ] Verify single run works: `python run.py dataset=iris model.L=2`
- [ ] Verify multirun works: `python run.py --multirun model.L=2,3`

### Phase 3: Full Config Migration
- [ ] Convert all 90 UCI configs to Hydra format
- [ ] Create experiment presets (`conf/experiment/`)
- [ ] Add SLURM launcher config
- [ ] Test SLURM submission on Titans cluster

### Phase 4: Validation & Cleanup
- [ ] Run full grid search on 1 dataset, compare with old results
- [ ] Add pytest tests for configs
- [ ] Update documentation
- [ ] Archive old `run_grid_search*.py` scripts
- [ ] Remove `jobs/generate_*.py` scripts

---

## Timeline Estimate

| Phase | Effort | Notes |
|-------|--------|-------|
| Phase 1: Setup | 2-4 hours | Config structure, dependencies |
| Phase 2: Runner | 4-8 hours | Unified runner, refactor training logic |
| Phase 3: Configs | 4-6 hours | Convert 90 JSON configs (can be scripted) |
| Phase 4: Testing | 2-4 hours | Validation, comparison tests |

**Total: 2-3 days of focused work**

---

## References

- [Hydra Documentation](https://hydra.cc/docs/intro/)
- [Hydra Submitit Launcher](https://hydra.cc/docs/plugins/submitit_launcher/)
- [Hydra Optuna Sweeper](https://hydra.cc/docs/plugins/optuna_sweeper/)
- [OmegaConf Documentation](https://omegaconf.readthedocs.io/)
