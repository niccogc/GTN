#!/bin/bash
#BSUB -q hpc
#BSUB -J "ntn_tnmlp_mushrooms"
#BSUB -W 8:00
#BSUB -n 6
#BSUB -R "rusage[mem=8GB]"
#BSUB -R "span[hosts=1]"
#BSUB -o logs/%J_tnmlp_mushrooms.out
#BSUB -e logs/%J_tnmlp_mushrooms.err

export HOME=/zhome/6b/e/212868
cd "$HOME/GTN"
source .venv/bin/activate

mkdir -p logs

export OMP_NUM_THREADS=6
export MKL_NUM_THREADS=6
export OPENBLAS_NUM_THREADS=6
export NUMEXPR_NUM_THREADS=6

echo "Running: model=tnml_p, dataset=mushrooms, experiment=uci_ntn_sweep"

python run.py --multirun \
    +experiment=uci_ntn_sweep \
    model=tnml_p \
    dataset=mushrooms
