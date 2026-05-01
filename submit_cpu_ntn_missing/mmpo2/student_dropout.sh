#!/bin/bash
#BSUB -q hpc
#BSUB -J "ntn_mmpo2_student_dropout"
#BSUB -W 8:00
#BSUB -n 6
#BSUB -R "rusage[mem=8GB]"
#BSUB -R "span[hosts=1]"
#BSUB -o logs/%J_mmpo2_student_dropout.out
#BSUB -e logs/%J_mmpo2_student_dropout.err

export HOME=/zhome/6b/e/212868
cd "$HOME/GTN"
source .venv/bin/activate

mkdir -p logs

export OMP_NUM_THREADS=6
export MKL_NUM_THREADS=6
export OPENBLAS_NUM_THREADS=6
export NUMEXPR_NUM_THREADS=6

echo "Running: model=mmpo2, dataset=student_dropout, experiment=uci_ntn_sweep"

python run.py --multirun \
    +experiment=uci_ntn_sweep \
    model=mmpo2 \
    dataset=student_dropout
