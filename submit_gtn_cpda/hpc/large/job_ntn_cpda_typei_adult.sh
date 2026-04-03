#!/bin/sh
#BSUB -q gpuv100
#BSUB -J ntn-cpda-typei-adult
#BSUB -W 24:00
#BSUB -n 4
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -R "rusage[mem=500MB]"
#BSUB -R "span[hosts=1]"
#BSUB -o logs/ntn-cpda-typei-adult_%J.out
#BSUB -e logs/ntn-cpda-typei-adult_%J.err
#BSUB -u nicci@dtu.dk

export HOME=/zhome/6b/e/212868
cd $HOME/GTN
source .venv/bin/activate

set -a && source $HOME/aim && set +a

python run.py --multirun +experiment=cpda_gtn_sweep model=cpda_typei dataset=adult
