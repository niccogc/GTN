#!/bin/bash
#BSUB -q hpc
#BSUB -J "ntn_tnmlp_adult[1-10]%10"
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
    "3 12 42"
    "3 12 10090"
    "3 12 32874"
    "3 12 47311"
    "3 12 47303"
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

echo "Running: model=tnml_p, dataset=adult, L=$L, bd=$BD, seed=$SEED"

python run.py \
    model=tnml_p \
    dataset=adult \
    model.L=$L \
    model.bond_dim=$BD \
    seed=$SEED
