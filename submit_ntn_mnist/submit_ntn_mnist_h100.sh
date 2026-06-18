#!/bin/bash
#BSUB -q gpuh100
#BSUB -J "ntn_cmpo2_mnist[1-2]"
#BSUB -W 24:00
#BSUB -n 10
#BSUB -R "rusage[mem=1GB]"
#BSUB -R "span[hosts=1]"
#BSUB -o logs/ntn_mpo2_%J.out
#BSUB -e logs/ntn_mpo2_%J.err
#BSUB -gpu "num=1:mode=exclusive_process"

cd ~/GTN
source .venv/bin/activate

mkdir -p logs


export OMP_NUM_THREADS=10
export MKL_NUM_THREADS=10
export OPENBLAS_NUM_THREADS=10
export NUMEXPR_NUM_THREADS=10
export NTN_MEMORY_CAP=60

if [[ $LSB_JOBINDEX == 1 ]]; then
  DATASET=_mnist
else
  DATASET=_fashion_mnist
fi

python run.py --multirun +experiment=_base_imgs dataset=$DATASET model=cmpo2 +use_suggested_batch=true 
