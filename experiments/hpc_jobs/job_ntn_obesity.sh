#!/bin/sh
#BSUB -q gpuv100
#BSUB -J ntn-obesity
#BSUB -W 12:00
#BSUB -n 4
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -R "rusage[mem=1GB]"
#BSUB -R "span[hosts=1]"
#BSUB -o logs/ntn-obesity_%J.out
#BSUB -e logs/ntn-obesity_%J.err
#BSUB -u nicci@dtu.dk

export HOME=/zhome/6b/e/212868

cd $HOME/GTN
source .venv/bin/activate

set -a
source $HOME/aim
set +a

python experiments/run_grid_search.py --config experiments/configs/uci_ntn_obesity.json 
