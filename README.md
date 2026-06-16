# Tensor Network Training (GTN)

A unified experiment runner for training **tensor-network machine-learning models**
on tabular (UCI / CSV) and image (MNIST, Fashion-MNIST, CIFAR) datasets.

Training is configured entirely through [Hydra](https://hydra.cc/) (`conf/`) and launched
through a single entry point: **`run.py`**.

---

## Table of Contents

- [Quick Start](#quick-start)
- [How `run.py` Works](#how-runpy-works)
- [Command-Line Flags & Overrides](#command-line-flags--overrides)
  - [Choosing model / dataset / trainer](#choosing-model--dataset--trainer)
  - [Overriding individual config values](#overriding-individual-config-values)
  - [Multirun / sweeps](#multirun--sweeps)
  - [Top-level run flags](#top-level-run-flags)
  - [Trainer flags](#trainer-flags)
- [Environment Variables](#environment-variables)
- [Outputs & Tracking](#outputs--tracking)
- [Adding a Dataset](#adding-a-dataset)
  - [The loader contract](#the-loader-contract)
  - [Option A: UCI dataset](#option-a-uci-dataset)
  - [Option B: Local CSV](#option-b-local-csv)
  - [Option C: Custom loader (your own format)](#option-c-custom-loader-your-own-format)
  - [Option D: Image dataset](#option-d-image-dataset)
- [Input Formalism (how data enters the network)](#input-formalism-how-data-enters-the-network)
- [Adding a Model](#adding-a-model)
  - [The model contract](#the-model-contract)
  - [Structure depending on the input encoding](#structure-depending-on-the-input-encoding)
  - [Step-by-step](#step-by-step)
  - [Adding a new input type / encoding](#adding-a-new-input-type--encoding)

---

## Quick Start

```bash
# Defaults: MPO2 model, iris dataset, NTN trainer
python run.py

# Override model & dataset
python run.py model=lmpo2 dataset=abalone

# GTN (gradient) trainer with a custom learning rate
python run.py trainer=gtn trainer.lr=0.01

# Image classification
python run.py model=cmpo2 dataset=mnist

# Sweep bond dimensions (Hydra multirun)
python run.py --multirun model.bond_dim=4,6,8
```

---

## How `run.py` Works

`run.py` is a Hydra application. Its configuration is composed from `conf/config.yaml`,
which selects one option from each of these config *groups*:

| Group     | Folder            | Default | Purpose                                  |
|-----------|-------------------|---------|------------------------------------------|
| `model`   | `conf/model/`     | `mpo2`  | Which tensor-network model to build      |
| `dataset` | `conf/dataset/`   | `iris`  | Which dataset to load                    |
| `trainer` | `conf/trainer/`   | `ntn`   | Which training algorithm to use          |

At runtime `run.py`:

1. Composes the config and seeds `torch` / `numpy` from `cfg.seed`.
2. (Optionally) looks up the best `L` / `bond_dim` from `conf/best_conf/` when `trainer.evaluate_test=true`.
3. Decides whether to **skip** the run if it already completed (see [Tracking](#outputs--tracking)).
4. Loads the dataset (tabular via `utils/dataset_loader.py`, image via `utils/image_dataset_loader.py`).
5. Builds the model from the registry in `run.py` (see [Adding a Model](#adding-a-model)).
6. Dispatches to the right training loop based on `trainer.type`:
   - `ntn`  → `run_ntn` (Newton-based, second-order)
   - `gtn`  → `run_gtn` (gradient descent via PyTorch autograd)
   - `dmrg` → `run_dmrg` (2-site DMRG, **TNML models only**)
   - `cnn`  → `run_cnn` (baseline CNN, image only)
   - image models route to `run_ntn_image` / `run_gtn_image`.
7. Writes `results.json` to the Hydra output directory and (optionally) appends a row to
   `runs_tracking.csv`.

### Available models

| Trainer support | Models |
|-----------------|--------|
| NTN / GTN       | `MPO2`, `LMPO2`, `MMPO2`, `CPDA` and their `*TypeI` variants (`MPO2TypeI`, `LMPO2TypeI`, `MMPO2TypeI`, `CPDATypeI`) |
| DMRG only       | `TNML_P`, `TNML_F` |
| GTN only        | `BosonMPS` |
| Image (NTN/GTN) | `CMPO2`, `CMPO3` |
| Image (CNN)     | `BaselineCNN` |

The corresponding Hydra config names are the lowercased file names in `conf/model/`
(e.g. `model=mpo2`, `model=lmpo2_typei`, `model=cmpo2`, `model=tnml_p`).

---

## Command-Line Flags & Overrides

`run.py` accepts standard Hydra override syntax. There are no `argparse`-style flags;
everything is `key=value`.

### Choosing model / dataset / trainer

```bash
python run.py model=<name> dataset=<name> trainer=<name>
```

- `model`   — any file in `conf/model/` (e.g. `mpo2`, `lmpo2`, `cpda`, `tnml_p`, `cmpo2`, `baseline_cnn`)
- `dataset` — any file in `conf/dataset/` (e.g. `iris`, `abalone`, `wine`, `adult`, ...)
- `trainer` — `ntn`, `gtn`, `dmrg`, or `cnn`

### Overriding individual config values

Any nested key can be overridden with dotted paths:

```bash
# Model architecture
python run.py model.L=5 model.bond_dim=12 model.init_strength=0.05

# Trainer hyperparameters
python run.py trainer.lr=0.001 trainer.n_epochs=500 trainer.ridge=2

# Dataset / batching
python run.py dataset.batch_size=1024

# Seed
python run.py seed=123
```

### Multirun / sweeps

Use `--multirun` (or `-m`) with comma-separated values to launch a grid:

```bash
# Sweep bond dimensions
python run.py --multirun model.bond_dim=4,6,8

# Multi-seed
python run.py --multirun seed=0,1,2,3,4

# Grid over multiple axes (Cartesian product)
python run.py --multirun model.L=3,4 model.bond_dim=8,12 seed=42,10090
```

Predefined sweep recipes live in `conf/experiment/`. Apply one with `+experiment=`:

```bash
python run.py --multirun +experiment=uci_ntn_sweep dataset=iris
python run.py --multirun +experiment=uci_gtn_sweep dataset=wine
python run.py --multirun +experiment=dmrg_sweep    dataset=abalone
```

### Top-level run flags

Defined in `conf/config.yaml`:

| Key                  | Default | Description |
|----------------------|---------|-------------|
| `seed`               | `42`    | Random seed for model init (data splits are fixed at seed 42 for reproducibility). |
| `skip_completed`     | `true`  | Skip a run if a successful/singular result already exists (checks `runs_tracking.csv` then `results.json`). |
| `update_tracking`    | `false` | Append the result to `runs_tracking.csv`. Keep `false` on clusters. |
| `save_model`         | `false` | Save the trained model (`model.joblib` for NTN, `model.pt` for GTN/CNN). |
| `use_suggested_batch`| (unset) | If `true`, override `dataset.batch_size` with a model-suggested value. |
| `data_dir`           | (unset) | Override the download/storage directory for image datasets. |

Example:

```bash
python run.py update_tracking=true save_model=true skip_completed=false
```

### Trainer flags

**NTN** (`conf/trainer/ntn.yaml`):

| Key                       | Default  | Description |
|---------------------------|----------|-------------|
| `trainer.n_epochs`        | `20`     | Number of sweeps. |
| `trainer.ridge`           | `5`      | Ridge / jitter regularization strength. |
| `trainer.ridge_decay`     | `0.25`   | Multiplicative decay of ridge per epoch. |
| `trainer.ridge_min`       | `0.0001` | Floor for the ridge value. |
| `trainer.adaptive_ridge`  | `false`  | Auto-increase jitter for ill-conditioned matrices. |
| `trainer.patience`        | `10`     | Early-stopping patience (`null` to disable). |
| `trainer.min_delta`       | `0.001`  | Minimum improvement counted as progress. |
| `trainer.train_selection` | `true`   | Use train quality as tiebreaker for model selection. |
| `trainer.evaluate_test`   | `false`  | Evaluate the test set every epoch (also loads best config). |

**GTN** (`conf/trainer/gtn.yaml`): adds `trainer.lr`, `trainer.optimizer`
(`adam`/`adamw`/`sgd`), `trainer.loss_fn` (`null`=auto, `mse`, `mae`, `huber`,
`cross_entropy`). `weight_decay` is derived as `2 * trainer.ridge`.

**DMRG** (`conf/trainer/dmrg.yaml`): adds `trainer.lr`, `trainer.max_bond`
(`null` = model bond dim), and `trainer.cutoff` (SVD truncation, default `1e-10`).
DMRG only supports `TNML_*` models.

**CNN** (`conf/trainer/cnn.yaml`): `trainer.lr`, `trainer.optimizer`,
`trainer.weight_decay`, `trainer.patience`, `trainer.min_delta`.

---

## Environment Variables

These control the **quimb / cotengra contraction strategy** used during tensor
contractions. They are read once at import time in `run.py`.

| Variable                   | Values                                                                 | Default | Description |
|----------------------------|------------------------------------------------------------------------|---------|-------------|
| `QUIMB_CONTRACT_STRATEGY`  | `greedy`, `auto`, `auto-hq`, `random-greedy`, `random-greedy-128`, `optimal` | (unset) | If set, use this fixed path-finding strategy instead of the cached hyper-optimizer. |
| `QUIMB_CONTRACT_MINIMIZE`  | `flops`, `write`, `combo`                                              | `flops` | What the default cached optimizer minimizes: `flops` = fastest execution, `write` = lowest memory, `combo` = balanced. |

Default behaviour (neither variable set): a `cotengra.ReusableHyperOptimizer` that
caches contraction paths to disk (`ctg_cache/`), minimizing FLOPs.

```bash
# Minimize GPU memory usage
QUIMB_CONTRACT_MINIMIZE=write python run.py

# Fast path-finding, no caching
QUIMB_CONTRACT_STRATEGY=greedy python run.py
```

> GPU vs CPU is selected automatically (`utils/device_utils.py`): CUDA is used when
> available, otherwise CPU. All tensors default to `float64`.

---

## Outputs & Tracking

- **Per-run output dir** — created by Hydra. The path templates are defined in the
  model configs, e.g. `outputs/<trainer>/<dataset>/<model>_rg<ridge>_init<init>/L<L>_bd<bond_dim>_seed<seed>/`.
  Each contains `results.json` (metrics, full `metrics_log`, resolved config, dataset info)
  and optional saved model artifacts.
- **Tracking CSV** — `runs_tracking.csv` (schema in `utils/tracking.py`). Written only when
  `update_tracking=true`. Used by `skip_completed=true` to avoid re-running completed
  experiments. Columns include `run_id`, model/dataset/trainer, `L`, `bond_dim`, `ridge`,
  `seed`, `success`, `singular`, `oom_error`, `val_quality`, etc.

---

## Adding a Dataset

There are four ways to add data. Options A–B reuse the built-in loaders; Option C is for
a completely custom format; Option D is for images. **In every case the data must
ultimately match the loader contract below**, because the training loops in `run.py`
expect a fixed dictionary shape.

### The loader contract

`run.py` (for tabular models) calls a single function:

```python
data, dataset_info = load_dataset(cfg.dataset.name, csv_path=..., task=...)
```

`data` **must** be a dict of `torch` tensors with exactly these keys:

| Key        | Shape                              | Notes |
|------------|------------------------------------|-------|
| `X_train`  | `(n_train, n_features)`            | `float64`, already scaled |
| `y_train`  | `(n_train, output_dim)`            | see target conventions below |
| `X_val`    | `(n_val, n_features)`              | |
| `y_val`    | `(n_val, output_dim)`             | |
| `X_test`   | `(n_test, n_features)`             | |
| `y_test`   | `(n_test, output_dim)`            | |

**Target (`y`) conventions** — handled automatically by `load_dataset`, but a custom
loader must reproduce them:

- **Regression:** `y` is `float64` and 2-D, i.e. shape `(n, output_dim)` (use
  `y.unsqueeze(1)` if you produced a 1-D vector).
- **Classification:** `y` is **one-hot** `float64` of shape `(n, n_classes)`. Class labels
  must be contiguous integers `0..n_classes-1` *before* one-hot encoding.

`dataset_info` is a metadata dict. The runner only needs it to be JSON-serialisable; the
built-in loaders fill in `name`, `source`, `n_features`, `n_train/val/test`, `task`, and
(for classification) `n_classes`.

> **X is never bias-padded by the loader.** The runner / `create_inputs` appends the bias
> column (a constant `1`) later, so `input_dim = n_features + 1` for the standard models.
> See [Input Formalism](#input-formalism-how-data-enters-the-network).

### Option A: UCI dataset

UCI datasets are fetched automatically via `ucimlrepo`.

1. **Register it** in `model/load_ucirepo.py` by adding a tuple to the `datasets` list:

   ```python
   datasets = [
       ...
       ("my_dataset", 999, "classification"),  # (name, UCI id, task)
   ]
   ```
   - `name`: the key you'll use as `dataset=my_dataset`.
   - `999`: the numeric UCI repository id.
   - `task`: `"classification"` or `"regression"`.
   - If the UCI metadata has the wrong target column, add an entry to
     `DATASETS_WITH_TARGET_FIX`.

2. **Create the Hydra config** `conf/dataset/my_dataset.yaml`:

   ```yaml
   # @package _global_
   defaults:
     - _base
     - size/small      # small | medium | large (controls batch_size + cluster resources)

   dataset:
     name: my_dataset   # MUST match the name registered above
     task: classification
   ```

3. **Run it:**

   ```bash
   python run.py dataset=my_dataset
   ```

Preprocessing (one-hot encoding capped at `dataset.cap` features, standard scaling,
fixed 70/15/15 train/val/test split at `random_state=42`, one-hot targets for
classification) is handled automatically in `model/_preproc.py`.

### Option B: Local CSV

For an arbitrary CSV file (convention: **last column = target**, all features numeric):

1. Place the file in `csvs/` (or use an absolute path).
2. Either:

   **Inline (no new config):**
   ```bash
   python run.py dataset=_csv dataset.csv_path=my_data.csv dataset.task=regression dataset.name=mydata
   ```

   **Or create `conf/dataset/mydata.yaml`:**
   ```yaml
   # @package _global_
   defaults:
     - _base
     - size/small

   dataset:
     name: mydata
     task: regression
     csv_path: my_data.csv   # relative to csvs/ or an absolute path
   ```
   then `python run.py dataset=mydata`.

CSV loading is implemented in `model/load_from_csv.py`.

### Option C: Custom loader (your own format)

Use this when your data is **not** UCI, not a last-column-target CSV, or needs
non-standard preprocessing (custom scaling, a fixed/predefined split, special target
handling, parquet/npz files, etc.).

You write a loader function and wire it into the dispatch inside
`utils/dataset_loader.py`. The function **must return the loader contract** described
above.

1. **Write the loader** (e.g. in `model/load_mydata.py`). The reusable helpers in
   `model/_preproc.py` do most of the work:

   ```python
   import pandas as pd
   from model._preproc import (
       split_data,        # fixed 70/15/15 split, random_state=42
       scale_dataframes,  # StandardScaler fit on train, applied to val/test
       to_tensors,        # DataFrames/Series -> float64 X tensors + typed y
   )

   def get_mydata(path, task="classification", device="cpu", cap=50):
       df = pd.read_parquet(path)              # or any source you like

       # 1) separate features / target however your format requires
       y = df["my_target"]
       X = df.drop(columns=["my_target"])

       # 2) (optional) make class labels contiguous 0..C-1 for classification
       if task == "classification":
           y = y.astype("category").cat.codes

       # 3) split  ->  4) scale  ->  5) to tensors
       X_tr, X_va, X_te, y_tr, y_va, y_te = split_data(X, y)
       X_tr, X_va, X_te = scale_dataframes(X_tr, X_va, X_te)
       return to_tensors(X_tr, X_va, X_te, y_tr, y_va, y_te, task, device=device)
   ```

   Notes on the helpers (all in `model/_preproc.py`):
   - `split_data(X, y, val_size=0.15, test_size=0.15, random_state=42)` → fixed,
     reproducible 70/15/15 split. Replace this call if you have a *predefined* split.
   - `scale_dataframes(X_train, X_val, X_test, num_cols=None)` → fits a `StandardScaler`
     on train only; pass `num_cols` to scale a subset.
   - `to_tensors(...)` → produces `float64` `X`, and `y` typed by `task` (regression →
     `float64` 2-D; classification → `long` indices, which `load_dataset` then one-hots).
   - `one_hot_with_cap(X, cap)` → one-hot encodes categorical columns, capping the total
     feature count at `cap` (drops the highest-cardinality columns first).

   If you bypass these helpers entirely, you are responsible for returning the six tensors
   with the dtypes/shapes from the [loader contract](#the-loader-contract).

2. **Register the loader** by branching inside `load_dataset` in
   `utils/dataset_loader.py`. The cleanest hook is the `dataset_name` (or a new
   `cfg.dataset.*` field you pass through):

   ```python
   # utils/dataset_loader.py, inside load_dataset(...)
   if dataset_name == "mydata":
       from model.load_mydata import get_mydata
       X_train, y_train, X_val, y_val, X_test, y_test = get_mydata(
           csv_path, task=task, device=device, cap=cap
       )
       source = "mydata"
       dataset_id = None
       _task = task
   # ... existing csv / uci branches ...
   ```

   The block after the branches already converts regression `y` to 2-D and classification
   `y` to one-hot, so returning `long` class indices from your loader is fine.

3. **Create the Hydra config** `conf/dataset/mydata.yaml`:

   ```yaml
   # @package _global_
   defaults:
     - _base
     - size/small

   dataset:
     name: mydata          # matched in load_dataset's branch
     task: classification
     csv_path: /abs/path/to/file.parquet   # reuse this field, or add your own
   ```

4. **Run it:** `python run.py dataset=mydata`.

### Option D: Image dataset

Image datasets (MNIST, Fashion-MNIST, CIFAR10/100) are loaded by
`utils/image_dataset_loader.py`. To add a new one:

1. Add its transforms in `get_image_transforms()` and metadata in `get_dataset_info()`
   inside `utils/image_dataset_loader.py`.
2. Create `conf/dataset/_my_image.yaml`:

   ```yaml
   # @package _global_
   dataset:
     name: MY_IMAGE        # MUST match the loader's expected name
     task: classification
     n_classes: 10
     image_size: 28
     channels: 1
     batch_size: 512
     n_train: null
     n_val: null
     n_test: null
   ```

3. Run with an image model: `python run.py model=cmpo2 dataset=_my_image`.

---

## Input Formalism (how data enters the network)

A model is a quimb `TensorNetwork` (`model.tn`) with named **open indices**. To run it,
the raw data tensor `X` is turned into one input `Tensor` per feature/site, and those are
contracted against the model's open indices. This is done by the `Inputs` builder
(`model/builder.py`) via the helper `create_inputs` (`model/utils.py`).

The contract has three parts that **must agree by name**:

1. **The model declares which open indices consume input** via two attributes:
   - `input_labels` — used by the `Inputs` builder to *name the index* on each input tensor.
   - `input_dims` — used by NTN/DMRG to identify the contracted input legs.
   - `output_dims` — the name(s) of the output leg (almost always `["out"]`).
2. **The builder creates input tensors** whose indices match `input_labels`.
3. **Contraction** pairs identical index names → the network produces an `(batch, out)` result.

There are three input styles in this repo; the style is determined by **how the model
sets `input_labels` and whether it sets an `encoding` attribute**:

### 1. Scalar inputs with a bias (default — MPO2 / LMPO2 / MMPO2 / CPDA)

Each feature is a **scalar** placed on its own site. A constant `1` bias feature is
appended, so a model with `L` sites consumes `L` scalar inputs and the dataset must have
`L - 1` real features (`input_dim = n_features + 1`).

- Model side: physical dimension of each site is `phys_dim = 1` per feature value, and
  `input_labels = ["x0", "x1", ..., "x{L-1}"]` (plain strings → one index per site).
- Runner side: `create_inputs(..., append_bias=True, encoding=None)` concatenates the
  bias column and builds one input tensor per site with index `s, x{i}`.

### 2. Encoded / feature-map inputs (TNML_P, TNML_F)

Each feature `x_i` is lifted to a **vector** via a feature map, so each site has a physical
dimension > 1 and there is **no bias term**. The model signals this by setting
`self.encoding`:

- `TNML_P` → `self.encoding = "polynomial"`; feature map `[1, x, x², …, x^degree]`
  (`phys_dim = degree + 1`, where `degree = L`).
- `TNML_F` → `self.encoding = "fourier"`; feature map `[cos(xπ/2), sin(xπ/2)]`
  (`phys_dim = 2`).

The runner detects `getattr(model, "encoding", None)` and calls `create_inputs` with that
encoding (no bias). Encoding functions live in `model/utils.py`
(`encode_polynomial`, `encode_fourier`).

### 3. Multi-source / structured inputs (image models, CMPO2/CMPO3)

A site can consume **more than one input index** (e.g. a pixel index *and* a patch index),
and inputs can come from multiple source tensors. This is expressed with the **explicit
list form** of `input_labels`:

```python
# CMPO2: each site i consumes two indices from input source 0
self.input_labels = [[0, (f"{i}_patches", f"{i}_pixels")] for i in range(L)]
```

The general `input_labels` grammar accepted by `Inputs` (`model/builder.py`):

| Form                         | Meaning |
|------------------------------|---------|
| `"x"` (str)                  | one index `x`, fed from input source `0` (or site `i` if 1-to-1). |
| `("p", "x")` (tuple)         | two indices on the same site, auto-mapped source. |
| `[src_idx, ("p", "x")]` (list) | **explicit**: take data from `inputs[src_idx]`, name the legs `p, x`. |

Every input tensor automatically gets a batch index (`"s"` by default) plus tags
`input_<inds>` and `I{i}`.

---

## Adding a Model

Adding a model requires (1) a model class that satisfies the **model contract**,
(2) registering it in `run.py`, and (3) a Hydra config.

### The model contract

A tabular model is a plain Python class whose `__init__` builds a quimb `TensorNetwork`
and exposes these attributes (consumed by `run.py`, `NTN`, `GTN`, `DMRG`):

| Attribute       | Type                     | Purpose |
|-----------------|--------------------------|---------|
| `self.tn`       | `quimb.tensor.TensorNetwork` | the trainable network with open input + output legs |
| `self.input_labels` | `list`               | how inputs are named/built (see [Input Formalism](#input-formalism-how-data-enters-the-network)) |
| `self.input_dims`   | `list[str]`          | the open input index names contracted by NTN/DMRG |
| `self.output_dims`  | `list[str]`          | the open output index name(s), usually `["out"]` |
| `self.bond_dim` | `int`                    | used as a DMRG default `max_bond` |
| `self.encoding` *(optional)* | `str`       | set to `"polynomial"`/`"fourier"` to request encoded inputs |
| `self.poly_degree` *(optional)* | `int`    | polynomial degree when `encoding == "polynomial"` |

Convention details that make contraction work:
- **Output leg must be named `"out"`** (matches `output_dims=["out"]`).
- **Input legs must be named to match `input_labels`** (e.g. `x{i}`, `{i}_in`, `{i}_pixels`).
- Tag each tensor uniquely (e.g. `Node{i}`, `{i}_MPS`) so input tensors created by the
  builder don't clash with model tensors.
- Mark any **non-trainable** tensors with the tag `"NT"` (see `MMPO2`'s mask) — NTN/GTN
  skip them during optimization.

> **TypeI models** are ensembles: instead of `tn`/`input_dims`/`input_labels` they expose
> `tns`, `input_dims_list`, `input_labels_list` (and shared `output_dims`). See
> `model/typeI/ntn_typeI.py`. They wrap the standard classes for `L = 1..max_sites`.

### Structure depending on the input encoding

Your model's tensor shapes depend on which input style you target (see
[Input Formalism](#input-formalism-how-data-enters-the-network)):

- **Scalar + bias (like MPO2):** each site's physical index has size `phys_dim` equal to
  the per-site input value count (the runner passes `phys_dim = n_features + 1`). The model
  sets `input_labels = ["x0", ..., "x{L-1}"]` (plain strings) and does **not** set
  `encoding`. Use this template:

  ```python
  # one MPS node per site, output leg "out" on output_site
  inds = (f"b{i-1}", f"x{i}", f"b{i}")   # bond, physical, bond
  if i == output_site:
      inds += ("out",)
  ```

- **Encoded feature map (like TNML):** set `self.encoding = "polynomial"` or `"fourier"`,
  size each physical index to the feature-map dimension (`degree+1` or `2`), and create
  **one site per feature** (no bias). The runner will feed encoded vectors automatically.

- **Structured / multi-index (like CMPO2):** put several open legs on a site and declare
  them with the explicit `input_labels` list form `[[src, (indA, indB)], ...]`. Register
  the model in `IMAGE_MODELS` and add construction logic in `create_image_model`.

### Step-by-step

#### 1. Implement the model class

Add a class under `model/standard/` (or `model/typeI/`, `model/image_models.py`) following
the contract above. Use `model/standard/MPO2_models.py` (`MPO2`) as the canonical template,
or `model/standard/TNML.py` for an encoded model.

Export it from the package `__init__.py` (e.g. `model/standard/__init__.py`).

#### 2. Register the model in `run.py`

Add the class to the appropriate registry near the top of `run.py`:

```python
NTN_MODELS = {           # usable by NTN and GTN trainers
    "MPO2": MPO2,
    ...
    "MyModel": MyModel,
}

GTN_TYPEI_MODELS = {...}  # GTN counterparts of *TypeI models
GTN_ONLY_MODELS = {...}   # models that only support GTN (e.g. BosonMPS)
IMAGE_MODELS = {...}      # image models (CMPO2, CMPO3, BaselineCNN)
IMAGE_GTN_MODELS = {...}  # GTN wrappers for image models
```

`create_model()` builds the model by calling the class with parameters from
`build_model_params()`. The standard params passed to a tabular model are:
`phys_dim`, `output_dim`, `output_site`, `init_strength`, `bond_dim`, `L`
(`*TypeI` models receive `max_sites` instead of `L`; TNML models receive the raw feature
count as `phys_dim`).

If your model needs **extra constructor arguments**, extend `build_model_params()` to read
them from `cfg.model.*` (and/or `create_model` / `create_image_model`). For example:

```python
# run.py, in build_model_params(...)
if cfg.model.name == "MyModel":
    params["my_extra"] = cfg.model.get("my_extra", 2)
```

#### 3. Create the Hydra config

Add `conf/model/mymodel.yaml`:

```yaml
# @package _global_
defaults:
  - _base          # provides L, bond_dim, output_site, init_strength defaults

model:
  name: MyModel    # MUST match the registry key in run.py
  # add any extra hyperparameters your model reads, e.g.:
  # my_extra: 2
```

The `_base` config (`conf/model/_base.yaml`) also defines the Hydra output-directory
template. Override `hydra.run.dir` / `hydra.sweep.subdir` in your model config if your
model has extra hyperparameters that should appear in the output path (see
`conf/model/lmpo2.yaml` and `conf/model/cmpo2.yaml` for examples).

#### 4. Run it

```bash
python run.py model=mymodel dataset=iris trainer=gtn
```

### Adding a new input type / encoding

To introduce a feature map beyond polynomial/Fourier (the third "different type of
inputs"):

1. **Write the encoding function** in `model/utils.py`, mapping `X (n, features)` →
   `(n, features, phys_dim)`:

   ```python
   def encode_mymap(X):              # e.g. [1, x, sin(x)]  -> phys_dim = 3
       return torch.stack([torch.ones_like(X), X, torch.sin(X)], dim=-1)
   ```

2. **Teach `create_inputs` (and `create_inputs_tnml`) about it** in `model/utils.py` by
   adding an `elif encoding == "mymap":` branch that calls your function and splits the
   result into one input tensor per feature (mirror the existing `"polynomial"` /
   `"fourier"` branches).

3. **Handle it in `run.py`'s GTN path** too: `run_gtn` encodes inputs inline (see the
   `if encoding == "polynomial": ... else: encode_fourier(...)` block) — add your branch
   there so gradient training matches NTN.

4. **Have your model request it** by setting `self.encoding = "mymap"` and sizing its
   physical indices to your `phys_dim`. The runner reads `getattr(model, "encoding", None)`
   and routes data through the new map automatically (no bias is appended for encoded
   inputs).

---

## Project Layout (relevant paths)

```
run.py                       # main entry point
conf/
  config.yaml                # top-level Hydra config
  model/                     # model configs (one per model)
  dataset/                   # dataset configs (+ size/ presets)
  trainer/                   # ntn / gtn / dmrg / cnn configs
  experiment/                # predefined sweep recipes
  best_conf/                 # per-model best L/bond_dim (used with evaluate_test)
model/
  standard/                  # MPO2, LMPO2, MMPO2, CPDA, TNML, BosonMPS
  typeI/                     # *TypeI model variants
  image_models.py            # CMPO2, CMPO3, BaselineCNN
  load_ucirepo.py            # UCI dataset registry + loader
  load_from_csv.py           # CSV loader
  _preproc.py                # shared preprocessing (split/scale/encode)
utils/
  dataset_loader.py          # tabular dataset entry point
  image_dataset_loader.py    # image dataset entry point
  device_utils.py            # CUDA/CPU selection
  tracking.py                # runs_tracking.csv schema + helpers
runs_tracking.csv            # experiment tracking log
outputs/                     # per-run results
```
