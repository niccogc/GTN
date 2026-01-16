#!/bin/sh
#BSUB -q gpuv100
#BSUB -J seoulBike-gtn-grid
#BSUB -W 12:00
#BSUB -n 8
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -R "rusage[mem=16GB]"
#BSUB -R "span[hosts=1]"
#BSUB -o logs/seoulBike-gtn-grid_%J.out
#BSUB -e logs/seoulBike-gtn-grid_%J.err
#BSUB -u nicci@dtu.dk

cd $HOME/GTN
source .venv/bin/activate

set -a
source $HOME/.ssh/aim
set +a

python experiments/run_grid_search_gtn.py --config experiments/configs/uci_gtn_seoulBike.json --output-dir results/seoulBike_gtn_grid
