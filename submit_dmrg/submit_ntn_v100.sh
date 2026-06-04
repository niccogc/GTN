#!/bin/bash
#BSUB -q gpuv100
#BSUB -J "ntn_grid_dmrg[1-20]%20"
#BSUB -W 12:00
#BSUB -n 6
#BSUB -R "rusage[mem=750MB]"
#BSUB -R "span[hosts=1]"
#BSUB -o logs/dmrg_%J_%I.out
#BSUB -e logs/dmrg_%J_%I.err
#BSUB -gpu "num=1:mode=exclusive_process"

cd "~/GTN"
source .venv/bin/activate
source missing.env

NUM_EXPERIMENTS=${#COMBINATIONS_DMRG[@]}

mkdir -p logs

IDV=$((LSB_JOBINDEX - 1))
read MODEL DATASET <<< "${COMBINATIONS_DMRG[$IDV]}"
echo "DMRG: $MODEL $DATASET"

echo "Task $LSB_JOBINDEX: Dataset=$DATASET, Model=$MODEL, Experiment=$EXPERIMENT"

export OMP_NUM_THREADS=6
export MKL_NUM_THREADS=6
export OPENBLAS_NUM_THREADS=6
export NUMEXPR_NUM_THREADS=6

python run.py --multirun \
    +experiment=dmrg_sweep \
    model=$MODEL \
    dataset=$DATASET
