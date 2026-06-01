#!/bin/bash
#BSUB -q hpc
#BSUB -J "ntn_grid_full[1-12]%20"
#BSUB -W 24:00
#BSUB -n 10
#BSUB -R "rusage[mem=8GB]"
#BSUB -R "span[hosts=1]"
#BSUB -o logs/%J_%I_cpu_newntn.out
#BSUB -e logs/%J_%I_cpu_newntn.err

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

export OMP_NUM_THREADS=10
export MKL_NUM_THREADS=10
export OPENBLAS_NUM_THREADS=10
export NUMEXPR_NUM_THREADS=10

python run.py --multirun \
    +experiment=$EXPERIMENT \
    model=$MODEL \
    dataset=$DATASET
