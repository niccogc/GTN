#!/bin/sh
#BSUB -q gpua100
#BSUB -J gtn-adult
#BSUB -W 24:00
#BSUB -n 4
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -R "rusage[mem=1GB]"
#BSUB -R "span[hosts=1]"
#BSUB -o logs/gtn-adult_%J.out
#BSUB -e logs/gtn-adult_%J.err
#BSUB -u nicci@dtu.dk

export HOME=/zhome/6b/e/212868

cd $HOME/GTN
source .venv/bin/activate

set -a
source $HOME/aim
set +a

python experiments/run_grid_search_gtn.py --config experiments/configs/uci_gtn_adult.json --output-dir results/gtn_adult
