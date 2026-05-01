#!/bin/bash
#BSUB -q hpc
#BSUB -J "ntn_mpo2typei_car_evaluation[1-16]%10"
#BSUB -W 1:00
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
    "3 12 47303"
    "4 4 42"
    "4 4 10090"
    "4 4 32874"
    "4 4 47311"
    "4 4 47303"
    "4 8 42"
    "4 8 10090"
    "4 8 32874"
    "4 8 47311"
    "4 8 47303"
    "4 12 42"
    "4 12 10090"
    "4 12 32874"
    "4 12 47311"
    "4 12 47303"
)

IDV=$((LSB_JOBINDEX - 1))
CONF=(${CONFIGS[$IDV]})

L=${CONF[0]}
BD=${CONF[1]}
SEED=${CONF[2]}

echo "Running: model=mpo2_typei, dataset=car_evaluation, L=$L, bd=$BD, seed=$SEED"

python run.py \
    model=mpo2_typei \
    dataset=car_evaluation \
    model.L=$L \
    model.bond_dim=$BD \
    seed=$SEED
