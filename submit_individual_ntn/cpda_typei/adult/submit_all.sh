#!/bin/bash
#BSUB -q hpc
#BSUB -J "ntn_cpdatypei_adult[1-30]%10"
#BSUB -W 3:00
#BSUB -n 6
#BSUB -R "rusage[mem=6GB]"
#BSUB -R "span[hosts=1]"
#BSUB -o logs/%J_%I.out
#BSUB -e logs/%J_%I.err

export HOME=/zhome/6b/e/212868
cd "$HOME/GTN"
source .venv/bin/activate
mkdir -p logs

export OMP_NUM_THREADS=6
export MKL_NUM_THREADS=6

# List of missing configs: "L bond_dim seed"
CONFIGS=(
    "3 8 42"
    "3 8 10090"
    "3 8 32874"
    "3 8 47311"
    "3 8 47303"
    "3 16 42"
    "3 16 10090"
    "3 16 32874"
    "3 16 47311"
    "3 16 47303"
    "3 32 42"
    "3 32 10090"
    "3 32 32874"
    "3 32 47311"
    "3 32 47303"
    "4 8 42"
    "4 8 10090"
    "4 8 32874"
    "4 8 47311"
    "4 8 47303"
    "4 16 42"
    "4 16 10090"
    "4 16 32874"
    "4 16 47311"
    "4 16 47303"
    "4 32 42"
    "4 32 10090"
    "4 32 32874"
    "4 32 47311"
    "4 32 47303"
)

IDV=$((LSB_JOBINDEX - 1))
CONF=(${CONFIGS[$IDV]})

L=${CONF[0]}
BD=${CONF[1]}
SEED=${CONF[2]}

echo "Running: model=cpda_typei, dataset=adult, L=$L, bd=$BD, seed=$SEED"

python run.py \
    model=cpda_typei \
    dataset=adult \
    model.L=$L \
    model.bond_dim=$BD \
    seed=$SEED
