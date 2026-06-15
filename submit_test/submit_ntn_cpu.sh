#!/bin/bash
#BSUB -q hpc
#BSUB -J "test_ntn[1-7]%20"
#BSUB -W 24:00
#BSUB -n 10
#BSUB -R "rusage[mem=4GB]"
#BSUB -R "span[hosts=1]"
#BSUB -o logs/%J_%I_test_ntn.out
#BSUB -e logs/%J_%I_test_ntn.err

cd "~/GTN"
source .venv/bin/activate
source missing_test.env

NUM_EXPERIMENTS=${#COMBINATIONS_NTN[@]}

mkdir -p logs

IDV=$((LSB_JOBINDEX - 1))
read MODEL DATASET <<< "${COMBINATIONS_NTN[$IDV]}"
echo "Test NTN: $MODEL $DATASET"

EXPERIMENT="test_ntn"

echo "Task $LSB_JOBINDEX: Dataset=$DATASET, Model=$MODEL, Experiment=$EXPERIMENT"

export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export NUMEXPR_NUM_THREADS=8

python run.py --multirun \
    +experiment=$EXPERIMENT \
    model=$MODEL \
    dataset=$DATASET
