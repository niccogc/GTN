#!/bin/bash
#BSUB -q gpuh100
#BSUB -J "test_ntn_missing[1-7]%20"
#BSUB -W 12:00
#BSUB -n 6
#BSUB -R "rusage[mem=750MB]"
#BSUB -R "span[hosts=1]"
#BSUB -o logs/%J_%I_test_ntn.out
#BSUB -e logs/%J_%I_test_ntn.err
#BSUB -gpu "num=1:mode=exclusive_process"

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

export OMP_NUM_THREADS=6
export MKL_NUM_THREADS=6
export OPENBLAS_NUM_THREADS=6
export NUMEXPR_NUM_THREADS=6

python run.py --multirun \
    +experiment=$EXPERIMENT \
    model=$MODEL \
    dataset=$DATASET
