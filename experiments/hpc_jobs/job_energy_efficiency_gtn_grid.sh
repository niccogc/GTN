#!/bin/sh
#BSUB -q gpuv100
#BSUB -J energy-efficiency-gtn-grid
#BSUB -W 3:00
#BSUB -n 8
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -R "rusage[mem=8GB]"
#BSUB -R "span[hosts=1]"
#BSUB -o logs/energy-efficiency-gtn-grid_%J.out
#BSUB -e logs/energy-efficiency-gtn-grid_%J.err
#BSUB -u nicci@dtu.dk

export HOME=/zhome/6b/e/212868
export AIM_REPO=$HOME/aim

cd $HOME/GTN
source .venv/bin/activate

set -a
source $HOME/.ssh/aim
set +a

python experiments/run_grid_search_gtn.py --config experiments/configs/uci_gtn_energy_efficiency.json --output-dir results/energy_efficiency_gtn_grid
