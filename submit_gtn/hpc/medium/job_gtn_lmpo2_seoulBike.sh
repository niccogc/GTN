#!/bin/sh
#BSUB -q gpuv100
#BSUB -J gtn-lmpo2-seoulBike
#BSUB -W 12:00
#BSUB -n 4
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -R "rusage[mem=500MB]"
#BSUB -R "span[hosts=1]"
#BSUB -o logs/gtn-lmpo2-seoulBike_%J.out
#BSUB -e logs/gtn-lmpo2-seoulBike_%J.err
#BSUB -u nicci@dtu.dk

export HOME=/zhome/6b/e/212868
cd $HOME/GTN
source .venv/bin/activate

set -a && source $HOME/aim && set +a

python run.py --multirun +experiment=uci_gtn_sweep model=lmpo2 dataset=seoulBike
