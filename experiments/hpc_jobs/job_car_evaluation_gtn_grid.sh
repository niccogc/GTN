#!/bin/sh
#BSUB -q gpuv100
#BSUB -J car-evaluation-gtn-grid
#BSUB -W 3:00
#BSUB -n 8
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -R "rusage[mem=8GB]"
#BSUB -R "span[hosts=1]"
#BSUB -o logs/car-evaluation-gtn-grid_%J.out
#BSUB -e logs/car-evaluation-gtn-grid_%J.err
#BSUB -u nicci@dtu.dk

cd $HOME/GTN
source .venv/bin/activate

set -a
source $HOME/.ssh/aim
set +a

python experiments/run_grid_search_gtn.py --config experiments/configs/uci_gtn_car_evaluation.json --output-dir results/car_evaluation_gtn_grid
