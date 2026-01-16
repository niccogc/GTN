#!/bin/sh
#BSUB -q gpua100
#BSUB -J student-dropout-gtn-grid
#BSUB -W 24:00
#BSUB -n 8
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -R "rusage[mem=32GB]"
#BSUB -R "span[hosts=1]"
#BSUB -o logs/student-dropout-gtn-grid_%J.out
#BSUB -e logs/student-dropout-gtn-grid_%J.err
#BSUB -u nicci@dtu.dk

cd $HOME/GTN
source .venv/bin/activate

set -a
source $HOME/.ssh/aim
set +a

python experiments/run_grid_search_gtn.py --config experiments/configs/uci_gtn_student_dropout.json --output-dir results/student_dropout_gtn_grid
