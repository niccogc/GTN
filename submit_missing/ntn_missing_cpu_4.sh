#!/bin/bash
#BSUB -q hpc
#BSUB -J "ntn_mmpo2typei_popularity"
#BSUB -W 12:00
#BSUB -n 16
#BSUB -R "rusage[mem=2387MB]"
#BSUB -R "span[hosts=1]"
#BSUB -o logs/%J.out
#BSUB -e logs/%J.err

source .venv/bin/activate

mkdir -p logs

echo "Model=mmpo2_typei, Dataset=popularity"

export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16
export OPENBLAS_NUM_THREADS=16
export NUMEXPR_NUM_THREADS=16

python run.py --multirun \
    +experiment=uci_ntn_sweep \
    model=mmpo2_typei \
    dataset=popularity
