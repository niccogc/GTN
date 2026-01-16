#!/bin/sh
#BSUB -q gpuv100
#BSUB -J bike-gtn-grid
#BSUB -W 12:00
#BSUB -n 8
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -R "rusage[mem=16GB]"
#BSUB -R "span[hosts=1]"
#BSUB -o logs/bike-gtn-grid_%J.out
#BSUB -e logs/bike-gtn-grid_%J.err
#BSUB -u nicci@dtu.dk

cd $HOME/GTN
source .venv/bin/activate

set -a
source $HOME/.ssh/aim
set +a

python experiments/run_grid_search_gtn.py --config experiments/configs/uci_gtn_bike.json --output-dir results/bike_gtn_grid
