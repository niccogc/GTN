#!/bin/bash
#BSUB -q hpc
#BSUB -J "time_comparison"
#BSUB -W 24:00
#BSUB -n 10
#BSUB -R "rusage[mem=8GB]"
#BSUB -R "span[hosts=1]"
#BSUB -o logs/%J_%I_cpu_newntn.out
#BSUB -e logs/%J_%I_cpu_newntn.err

cd "~/GTN"
source .venv/bin/activate

mkdir -p logs

export OMP_NUM_THREADS=10
export MKL_NUM_THREADS=10
export OPENBLAS_NUM_THREADS=10
export NUMEXPR_NUM_THREADS=10

python run.py --multirun \
    +experiment=time_comparison \
