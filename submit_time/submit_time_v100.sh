#!/bin/bash
#BSUB -q gpuv100
#BSUB -J "time"
#BSUB -W 2:00
#BSUB -n 12
#BSUB -R "rusage[mem=750MB]"
#BSUB -R "span[hosts=1]"
#BSUB -o logs/time_%J.out
#BSUB -e logs/time_%J.err
#BSUB -gpu "num=1:mode=exclusive_process"

cd "~/GTN"
source .venv/bin/activate

mkdir -p logs

export OMP_NUM_THREADS=12
export MKL_NUM_THREADS=12
export OPENBLAS_NUM_THREADS=12
export NUMEXPR_NUM_THREADS=12

python run.py --multirun \
    +experiment=time_comparison \
