#!/bin/sh
#BSUB -q gpuv100
#BSUB -J breast-gtn-lmpo2-reduction
#BSUB -W 3:00
#BSUB -n 8
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -R "rusage[mem=8GB]"
#BSUB -R "span[hosts=1]"
#BSUB -o logs/breast-gtn-lmpo2-reduction_%J.out
#BSUB -e logs/breast-gtn-lmpo2-reduction_%J.err
#BSUB -u nicci@dtu.dk

cd $HOME/GTN
source .venv/bin/activate

set -a
source $HOME/.ssh/aim
set +a

python experiments/run_grid_search_gtn.py --config experiments/configs/uci_gtn_lmpo2_breast.json --output-dir results/breast_gtn_lmpo2_reduction
