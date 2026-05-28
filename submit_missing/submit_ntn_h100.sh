#!/bin/bash
#BSUB -q gpuh100
#BSUB -J "ntn_grid_missing[1-12]%20"
#BSUB -W 20:00
#BSUB -n 6
#BSUB -R "rusage[mem=750MB]"
#BSUB -R "span[hosts=1]"
#BSUB -o logs/%J_%I_gpuh_newntn.out
#BSUB -e logs/%J_%I_gpuh_newntn.err
#BSUB -gpu "num=1:mode=exclusive_process"

cd "~/GTN"
source .venv/bin/activate
source missing.env

NUM_EXPERIMENTS=${#COMBINATIONS_NTN[@]}

mkdir -p logs

IDV=$((LSB_JOBINDEX - 1))
read MODEL DATASET <<< "${COMBINATIONS_NTN[$IDV]}"
echo "NTN: $MODEL $DATASET"

if [[ "$MODEL" == "cpda" || "$MODEL" == "cpda_typei" ]]; then
    EXPERIMENT="cpda_ntn_sweep"
else
    EXPERIMENT="uci_ntn_sweep"
fi

echo "Task $LSB_JOBINDEX: Dataset=$DATASET, Model=$MODEL, Experiment=$EXPERIMENT"

export OMP_NUM_THREADS=6
export MKL_NUM_THREADS=6
export OPENBLAS_NUM_THREADS=6
export NUMEXPR_NUM_THREADS=6

python run.py --multirun \
    +experiment=$EXPERIMENT \
    model=$MODEL \
    dataset=$DATASET
