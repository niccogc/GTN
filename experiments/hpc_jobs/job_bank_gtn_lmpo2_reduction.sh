#!/bin/sh
#BSUB -q gpua100
#BSUB -J bank-gtn-lmpo2-reduction
#BSUB -W 24:00
#BSUB -n 8
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -R "rusage[mem=32GB]"
#BSUB -R "span[hosts=1]"
#BSUB -o logs/bank-gtn-lmpo2-reduction_%J.out
#BSUB -e logs/bank-gtn-lmpo2-reduction_%J.err
#BSUB -u nicci@dtu.dk

cd $HOME/GTN
source .venv/bin/activate

set -a
source $HOME/.ssh/aim
set +a

python experiments/run_grid_search_gtn.py --config experiments/configs/uci_gtn_lmpo2_bank.json --output-dir results/bank_gtn_lmpo2_reduction
