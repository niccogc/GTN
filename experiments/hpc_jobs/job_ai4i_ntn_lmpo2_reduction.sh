#!/bin/sh
#BSUB -q gpuv100
#BSUB -J ai4i-ntn-lmpo2-reduction
#BSUB -W 12:00
#BSUB -n 8
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -R "rusage[mem=16GB]"
#BSUB -R "span[hosts=1]"
#BSUB -o logs/ai4i-ntn-lmpo2-reduction_%J.out
#BSUB -e logs/ai4i-ntn-lmpo2-reduction_%J.err
#BSUB -u nicci@dtu.dk

cd $HOME/GTN
source .venv/bin/activate

set -a
source $HOME/.ssh/aim
set +a

python experiments/run_grid_search.py --config experiments/configs/uci_ntn_lmpo2_ai4i.json 
