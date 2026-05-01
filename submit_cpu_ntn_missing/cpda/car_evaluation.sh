#!/bin/bash
#BSUB -q hpc
#BSUB -J "ntn_cpda_car_evaluation"
#BSUB -W 8:00
#BSUB -n 6
#BSUB -R "rusage[mem=4GB]"
#BSUB -R "span[hosts=1]"
#BSUB -o logs/%J_cpda_car_evaluation.out
#BSUB -e logs/%J_cpda_car_evaluation.err

export HOME=/zhome/6b/e/212868
cd "$HOME/GTN"
source .venv/bin/activate

mkdir -p logs

export OMP_NUM_THREADS=6
export MKL_NUM_THREADS=6
export OPENBLAS_NUM_THREADS=6
export NUMEXPR_NUM_THREADS=6

echo "Running: model=cpda, dataset=car_evaluation, experiment=cpda_ntn_sweep"

python run.py --multirun \
    +experiment=cpda_ntn_sweep \
    model=cpda \
    dataset=car_evaluation
