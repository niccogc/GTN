#!/bin/sh
#BSUB -q gpuv100
#BSUB -J car-evaluation-ntn-lmpo2-reduction
#BSUB -W 3:00
#BSUB -n 8
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -R "rusage[mem=8GB]"
#BSUB -R "span[hosts=1]"
#BSUB -o logs/car-evaluation-ntn-lmpo2-reduction_%J.out
#BSUB -e logs/car-evaluation-ntn-lmpo2-reduction_%J.err
#BSUB -u nicci@dtu.dk

cd $HOME/GTN
source .venv/bin/activate

set -a
source $HOME/.ssh/aim
set +a

python experiments/run_grid_search.py --config experiments/configs/uci_ntn_lmpo2_car_evaluation.json 
