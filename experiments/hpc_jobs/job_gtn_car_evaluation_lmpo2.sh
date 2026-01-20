#!/bin/sh
#BSUB -q gpuv100
#BSUB -J gtn-car-evaluation-lmpo2
#BSUB -W 3:00
#BSUB -n 8
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -R "rusage[mem=8GB]"
#BSUB -R "span[hosts=1]"
#BSUB -o logs/gtn-car-evaluation-lmpo2_%J.out
#BSUB -e logs/gtn-car-evaluation-lmpo2_%J.err
#BSUB -u nicci@dtu.dk

export HOME=/zhome/6b/e/212868

cd $HOME/GTN
source .venv/bin/activate

set -a
source $HOME/aim
set +a

python experiments/run_grid_search_gtn.py --config experiments/configs/uci_gtn_car_evaluation_lmpo2.json --output-dir results/gtn_car_evaluation_lmpo2
