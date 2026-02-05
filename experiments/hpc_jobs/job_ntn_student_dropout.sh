#!/bin/sh
#BSUB -q p1
#BSUB -J ntn-student-dropout
#BSUB -W 24:00
#BSUB -n 4
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -R "rusage[mem=500MB]"
#BSUB -R "span[hosts=1]"
#BSUB -o logs/ntn-student-dropout_%J.out
#BSUB -e logs/ntn-student-dropout_%J.err
#BSUB -u nicci@dtu.dk

export HOME=/zhome/6b/e/212868

cd $HOME/GTN
source .venv/bin/activate

set -a
source $HOME/aim
set +a

python experiments/run_grid_search.py --config experiments/configs/uci_ntn_student_dropout.json 
