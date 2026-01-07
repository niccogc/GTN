#!/bin/bash
# Bash wrapper for MPO2 training with environment variable configuration

# Task type
export TASK="${TASK:-regression}"

# Dataset
export DATASET="${DATASET:-california_housing}"
export N_SAMPLES="${N_SAMPLES:-500}"

# Model architecture
export L="${L:-3}"
export BOND_DIM="${BOND_DIM:-6}"
export OUTPUT_SITE="${OUTPUT_SITE:-1}"
export INIT_STRENGTH="${INIT_STRENGTH:-0.1}"

# Training
export BATCH_SIZE="${BATCH_SIZE:-100}"
export N_EPOCHS="${N_EPOCHS:-10}"
export JITTER_START="${JITTER_START:-0.05}"
export JITTER_DECAY="${JITTER_DECAY:-0.9}"
export JITTER_MIN="${JITTER_MIN:-1e-6}"

# Multi-seed
export SEEDS="${SEEDS:-0,1,2,3,4}"

# Output
export OUTPUT_FILE="${OUTPUT_FILE:-results_mpo2.json}"
export VERBOSE="${VERBOSE:-false}"

# Tracking
export TRACKER_BACKEND="${TRACKER_BACKEND:-file}"
export TRACKER_DIR="${TRACKER_DIR:-experiment_logs}"
export AIM_REPO="${AIM_REPO}"
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-mpo2_experiment}"

# Run training
python experiments/train_mpo2.py
