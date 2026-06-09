#!/bin/bash
#BSUB -q hpc
#BSUB -J "test_gtn_missing[1-94]%20"
#BSUB -W 24:00
#BSUB -n 6
#BSUB -R "rusage[mem=1GB]"
#BSUB -R "span[hosts=1]"
#BSUB -o logs/%J_%I_test_gtn.out
#BSUB -e logs/%J_%I_test_gtn.err

cd "~/GTN"
source .venv/bin/activate
source missing_test.env

NUM_EXPERIMENTS=${#COMBINATIONS_GTN[@]}

mkdir -p logs

IDV=$((LSB_JOBINDEX - 1))
read MODEL DATASET <<< "${COMBINATIONS_GTN[$IDV]}"
echo "Test GTN: $MODEL $DATASET"

EXPERIMENT="test_gtn"

echo "Task $LSB_JOBINDEX: Dataset=$DATASET, Model=$MODEL, Experiment=$EXPERIMENT"

export OMP_NUM_THREADS=6
export MKL_NUM_THREADS=6
export OPENBLAS_NUM_THREADS=6
export NUMEXPR_NUM_THREADS=6

python run.py --multirun \
    +experiment=$EXPERIMENT \
    model=$MODEL \
    dataset=$DATASET
