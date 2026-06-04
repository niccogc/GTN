#!/bin/bash
#BSUB -q gpuv100
#BSUB -J "cnn_mnist"
#BSUB -W 24:00
#BSUB -n 4
#BSUB -R "rusage[mem=2GB]"
#BSUB -R "span[hosts=1]"
#BSUB -o logs/cnn_mnist_%J.out
#BSUB -e logs/cnn_mnist_%J.err
#BSUB -gpu "num=1:mode=exclusive_process"

cd ~/GTN
source .venv/bin/activate

mkdir -p logs

export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4

python run.py --multirun +experiment=cnn_test
