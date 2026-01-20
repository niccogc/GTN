#!/bin/sh
#BSUB -q gpua100
#BSUB -J cmpo2-cifar100
#BSUB -W 24:00
#BSUB -n 8
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -R "rusage[mem=32GB]"
#BSUB -R "span[hosts=1]"
#BSUB -o logs/cmpo2-cifar100_%J.out
#BSUB -e logs/cmpo2-cifar100_%J.err
#BSUB -u nicci@dtu.dk

export HOME=/zhome/6b/e/212868

cd $HOME/GTN
source .venv/bin/activate

set -a
source $HOME/aim
set +a

python experiments_images/run_grid_search_cmpo2.py --config experiments_images/configs/cmpo2_cifar100.json --output-dir results/cmpo2_cifar100
