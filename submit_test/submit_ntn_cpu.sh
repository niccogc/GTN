#!/bin/bash
#BSUB -q hpc
#BSUB -J "test_ntn[1-5]%20"
#BSUB -o logs/%J_%I_test_ntn.out
#BSUB -e logs/%J_%I_test_ntn.err
#BSUB -W 24:00
#BSUB -n 10
#BSUB -R "rusage[mem=2GB]"
#BSUB -R "span[hosts=1]"

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

export OMP_NUM_THREADS=10
export MKL_NUM_THREADS=10
export OPENBLAS_NUM_THREADS=10
export NUMEXPR_NUM_THREADS=10

python run.py --multirun \
    +experiment=$EXPERIMENT \
    model=$MODEL \
    dataset=$DATASET \
    use_suggested_batch=true
