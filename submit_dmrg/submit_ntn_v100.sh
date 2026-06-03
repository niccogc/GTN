#!/bin/bash
#BSUB -q gpuv100
#BSUB -J "ntn_grid_missing[1-42]%20"
#BSUB -W 20:00
#BSUB -n 6
#BSUB -R "rusage[mem=750MB]"
#BSUB -R "span[hosts=1]"
#BSUB -o logs/%J_%I_gpuh_newntn.out
#BSUB -e logs/%J_%I_gpuh_newntn.err
#BSUB -gpu "num=1:mode=exclusive_process"

cd "~/GTN"
source .venv/bin/activate
source dmrg.env

NUM_EXPERIMENTS=${#COMBINATIONS_NTN[@]}

mkdir -p logs

IDV=$((LSB_JOBINDEX - 1))
read MODEL DATASET <<< "${COMBINATIONS_NTN[$IDV]}"
echo "NTN: $MODEL $DATASET"

echo "Task $LSB_JOBINDEX: Dataset=$DATASET, Model=$MODEL, Experiment=$EXPERIMENT"

export OMP_NUM_THREADS=6
export MKL_NUM_THREADS=6
export OPENBLAS_NUM_THREADS=6
export NUMEXPR_NUM_THREADS=6

python run.py --multirun \
    +experiment=dmrg_sweep \
    model=$MODEL \
    dataset=$DATASET
