#!/bin/sh
#BSUB -q gpuv100
#BSUB -J ntn-concrete-lmpo2
#BSUB -W 3:00
#BSUB -n 8
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -R "rusage[mem=16GB]"
#BSUB -R "span[hosts=1]"
#BSUB -o logs/ntn-concrete-lmpo2_%J.out
#BSUB -e logs/ntn-concrete-lmpo2_%J.err
#BSUB -u nicci@dtu.dk

export HOME=/zhome/6b/e/212868

cd $HOME/GTN
source .venv/bin/activate

set -a
source $HOME/aim
set +a

python experiments/run_grid_search.py --config experiments/configs/uci_ntn_concrete_lmpo2.json 
