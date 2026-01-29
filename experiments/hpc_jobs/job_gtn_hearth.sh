#!/bin/sh
#BSUB -q gpuv100
#BSUB -J gtn-hearth
#BSUB -W 6:00
#BSUB -n 4
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -R "rusage[mem=2GB]"
#BSUB -R "span[hosts=1]"
#BSUB -o logs/gtn-hearth_%J.out
#BSUB -e logs/gtn-hearth_%J.err
#BSUB -u nicci@dtu.dk

export HOME=/zhome/6b/e/212868

cd $HOME/GTN
source .venv/bin/activate

set -a
source $HOME/aim
set +a

python experiments/run_grid_search_gtn.py --config experiments/configs/uci_gtn_hearth.json --output-dir results/gtn_hearth
