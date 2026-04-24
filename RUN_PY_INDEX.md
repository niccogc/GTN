# run.py Analysis - Complete Documentation Index

## Overview

This directory contains comprehensive documentation of `run.py`, the unified experiment runner for tensor network models. The analysis covers all aspects of the codebase from configuration to model forward passes.

## Documents

### 1. **ANALYSIS_SUMMARY.txt** (215 lines)
**Quick reference guide with key findings**

- Executive summary of all 5 key areas
- Key findings checklist
- Execution flow diagram
- Important notes and gotchas
- Quick start examples
- File locations

**Best for:** Getting a quick overview, finding specific information

---

### 2. **RUN_PY_ANALYSIS.md** (579 lines)
**Comprehensive technical analysis**

#### Section 1: Model Instantiation & Configuration
- Hydra configuration system
- Config hierarchy and defaults
- Model instantiation flow
- Model registry (10 models)
- Parameter building logic
- Configuration examples for each model type

#### Section 2: Input Preparation
- Dataset loading from UCI ML Repository
- Data structure and shapes
- Input preparation for NTN (quimb Inputs class)
- Input preparation for GTN (PyTorch DataLoaders)
- Inputs class details
- Encoding options (polynomial, Fourier)

#### Section 3: Training/Testing
- Trainer selection (NTN vs GTN)
- NTN training (Newton-based, 20 epochs)
- GTN training (gradient-based, 1000 epochs)
- Loss functions and metrics
- Early stopping and model selection
- Error handling (singular matrices, OOM)

#### Section 4: Available Datasets
- Complete dataset registry (21 datasets)
- **IMPORTANT: No small regression datasets!**
- Smallest regression: concrete (165 samples)
- Small classification datasets
- Dataset loading details
- Feature encoding and preprocessing

#### Section 5: Model Type Specification
- Command-line overrides
- Configuration file approach
- Model-specific parameters
- Trainer-specific parameters
- Dataset-specific parameters
- Grid search examples

#### Section 6: Full Execution Flow
- Complete step-by-step flow diagram
- Model forward pass details (NTN)
- Model forward pass details (GTN)
- Tensor network structure visualization

#### Section 7-8: Quick Reference & Key Insights
- Common commands
- Model types and characteristics
- Training methods comparison
- Input handling differences
- Dataset split strategy
- No small regression datasets note

**Best for:** Understanding the complete system, detailed technical reference

---

### 3. **RUN_PY_CODE_EXAMPLES.md** (847 lines)
**Practical code examples and deep dives**

#### Section 1: Model Instantiation Examples
- **Example 1:** MPO2 (Simple MPS)
  - Configuration
  - Instantiation code
  - Resulting tensor network structure
  
- **Example 2:** LMPO2 (With Dimensionality Reduction)
  - Two-layer architecture
  - Data flow diagram
  
- **Example 3:** TNML_P (Polynomial Encoding)
  - Feature encoding process
  - Input structure
  
- **Example 4:** MPO2TypeI (Variable Sites Ensemble)
  - Ensemble structure

#### Section 2: Input Preparation Examples
- **Example 1:** Standard Model Input (NTN)
  - Raw data shapes
  - Bias term appending
  - Batch structure
  
- **Example 2:** TNML Model Input (NTN)
  - Polynomial encoding
  - Multi-feature input structure
  
- **Example 3:** GTN Input Preparation
  - Standard model DataLoader
  - TNML model DataLoader

#### Section 3: Training Examples
- **Example 1:** NTN Training Loop
  - Ridge schedule
  - Epoch callback
  - Training output example
  
- **Example 2:** GTN Training Loop
  - Optimizer setup
  - Training/evaluation phases
  - Early stopping logic
  - Training output example

#### Section 4: Command-Line Usage
- Basic examples
- Hyperparameter tuning
- Grid search (multirun)
- Advanced examples

#### Section 5: Result Inspection
- Results JSON structure
- Metrics log format
- Config and dataset_info
- Python code for analyzing results
- Plotting training curves

**Best for:** Practical implementation, copy-paste code examples, debugging

---

## Quick Navigation

### I want to understand...

**How models are created:**
- Start: ANALYSIS_SUMMARY.txt (Key Finding #1)
- Deep dive: RUN_PY_ANALYSIS.md Section 1
- Code: RUN_PY_CODE_EXAMPLES.md Section 1

**How inputs are prepared:**
- Start: ANALYSIS_SUMMARY.txt (Key Finding #2)
- Deep dive: RUN_PY_ANALYSIS.md Section 2
- Code: RUN_PY_CODE_EXAMPLES.md Section 2

**How training works:**
- Start: ANALYSIS_SUMMARY.txt (Key Finding #3)
- Deep dive: RUN_PY_ANALYSIS.md Section 3
- Code: RUN_PY_CODE_EXAMPLES.md Section 3

**What datasets are available:**
- Start: ANALYSIS_SUMMARY.txt (Key Finding #4)
- Deep dive: RUN_PY_ANALYSIS.md Section 4
- Note: **NO small regression datasets exist!**

**How to run experiments:**
- Start: ANALYSIS_SUMMARY.txt (Quick Start Examples)
- Deep dive: RUN_PY_ANALYSIS.md Section 5
- Code: RUN_PY_CODE_EXAMPLES.md Section 4

**The complete execution flow:**
- Start: ANALYSIS_SUMMARY.txt (Execution Flow Summary)
- Deep dive: RUN_PY_ANALYSIS.md Section 6
- Diagram: Full flow with all steps

---

## Key Findings Summary

### 1. Model Instantiation
- **10 models available:** MPO2, LMPO2, MMPO2, CPDA, + TypeI variants + TNML variants
- **Registry-based:** Models looked up from `NTN_MODELS` dictionary
- **Dynamic parameters:** Built from config based on dataset dimensions
- **Special handling:** TypeI (variable sites), TNML (feature encoding)

### 2. Input Preparation
- **Standard models:** Bias term appended to features
- **TNML models:** Features encoded (polynomial or Fourier)
- **NTN:** Uses quimb Inputs class with pre-computed batches
- **GTN:** Uses PyTorch DataLoaders
- **Classification:** Targets converted to one-hot
- **Regression:** Targets kept as continuous

### 3. Training Methods
- **NTN:** Newton-based, 20 epochs, ridge regularization with decay
- **GTN:** Gradient-based, 1000 epochs, standard optimizers
- **Both:** Early stopping, model selection, error handling

### 4. Datasets
- **21 total:** 11 regression, 10 classification
- **⚠️ CRITICAL:** NO small regression datasets!
- **Smallest regression:** concrete (165 samples) - Medium
- **Small classification:** iris, hearth, breast, wine, car_evaluation
- **Fixed splits:** 70/15/15 (seed=42)

### 5. Configuration
- **Command-line:** `python run.py model=lmpo2 dataset=abalone trainer=gtn`
- **Config files:** Create `conf/experiment/my_experiment.yaml`
- **Grid search:** `python run.py --multirun model.bond_dim=4,6,8`
- **All parameters:** Overridable from command line

---

## Important Notes

### ⚠️ No Small Regression Datasets
The current dataset collection has **NO small regression datasets**. The smallest regression dataset is "concrete" with 165 samples (classified as Medium). If you need small datasets, use classification datasets instead.

### Dataset Splits
- Train/val/test splits are **FIXED** (seed=42) for reproducibility
- Experiment seed controls **model initialization**, not data splits
- This isolates model variance from data variance

### Bias Term
- **Standard models:** Bias appended (input_dim = raw_features + 1)
- **TNML models:** NO bias (uses feature encoding instead)

### Feature Encoding
- **Standard:** Raw features used directly
- **TNML_P:** Polynomial [1, x, x², ..., x^degree]
- **TNML_F:** Fourier [cos(x·π/2), sin(x·π/2)]

### Ridge Regularization
- **NTN:** Decays per epoch (ridge_decay=0.25)
- **GTN:** weight_decay = 2 × ridge (to match NTN)

---

## File Structure

```
/home/nicci/Desktop/remote/GTNbos/
├── run.py                          # Main script (727 lines)
├── RUN_PY_INDEX.md                 # This file
├── ANALYSIS_SUMMARY.txt            # Quick reference
├── RUN_PY_ANALYSIS.md              # Comprehensive analysis
├── RUN_PY_CODE_EXAMPLES.md         # Code examples
├── conf/                           # Configuration files
│   ├── config.yaml
│   ├── model/                      # 10 model configs
│   ├── trainer/                    # 2 trainer configs
│   └── dataset/                    # 21 dataset configs
├── model/                          # Model implementations
│   ├── standard/                   # Standard models
│   ├── typeI/                      # TypeI variants
│   ├── base/                       # NTN.py, GTN.py
│   └── builder.py                  # Inputs class
└── utils/                          # Utilities
    ├── dataset_loader.py
    ├── device_utils.py
    └── tracking.py
```

---

## Quick Start

```bash
# Default: MPO2 + iris + NTN
python run.py

# Change model
python run.py model=lmpo2

# Change dataset
python run.py dataset=wine

# Change trainer
python run.py trainer=gtn

# Combine
python run.py model=lmpo2 dataset=concrete trainer=gtn

# Grid search
python run.py --multirun model.bond_dim=4,6,8

# Custom hyperparameters
python run.py model=lmpo2 model.reduction_factor=0.5 trainer=gtn trainer.lr=0.01
```

---

## Document Statistics

| Document | Lines | Size | Purpose |
|----------|-------|------|---------|
| ANALYSIS_SUMMARY.txt | 215 | 8.6K | Quick reference |
| RUN_PY_ANALYSIS.md | 579 | 18K | Comprehensive analysis |
| RUN_PY_CODE_EXAMPLES.md | 847 | 22K | Code examples |
| **Total** | **1,641** | **48.6K** | Complete documentation |

---

## How to Use These Documents

1. **First time?** Start with ANALYSIS_SUMMARY.txt
2. **Need details?** Read RUN_PY_ANALYSIS.md
3. **Want code?** Check RUN_PY_CODE_EXAMPLES.md
4. **Quick lookup?** Use this index

---

## Contact & Questions

For questions about specific sections:
- **Configuration:** See RUN_PY_ANALYSIS.md Section 1
- **Data flow:** See RUN_PY_ANALYSIS.md Section 2
- **Training:** See RUN_PY_ANALYSIS.md Section 3
- **Datasets:** See RUN_PY_ANALYSIS.md Section 4
- **Commands:** See RUN_PY_CODE_EXAMPLES.md Section 4

---

*Last updated: 2024-04-24*
*Analysis of: run.py (727 lines)*
*Total documentation: 1,641 lines across 3 documents*
