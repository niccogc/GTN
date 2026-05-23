#!/bin/bash
#SBATCH --job-name=gtn_sweep
#SBATCH --output=/home/nicci/GTN/logs/gtn_%A_%a.out
#SBATCH --error=/home/nicci/GTN/logs/gtn_%A_%a.err
#SBATCH --partition=titans
#SBATCH --time=24:00:00
#SBATCH --mem=4gb
#SBATCH --gres=gpu
#SBATCH --cpus-per-task=8
#SBATCH --array=1-3%3

export HOME=/home/nicci
cd $HOME/GTN
source $(conda info --base)/etc/profile.d/conda.sh
conda activate gtn

mkdir -p logs
source missing.env
IDV=$((SLURM_ARRAY_TASK_ID - 1))

read MODEL DATASET <<< "${COMBINATIONS_GTN[$IDV]}"
echo "GTN: $MODEL $DATASET"

if [[ "$MODEL" == "cpda" || "$MODEL" == "cpda_typei" ]]; then
    EXPERIMENT="cpda_gtn_sweep"
else
    EXPERIMENT="uci_gtn_sweep"
fi

echo "Task $SLURM_ARRAY_TASK_ID: Dataset=$DATASET, Model=$MODEL, Experiment=$EXPERIMENT"

export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export NUMEXPR_NUM_THREADS=8

python run.py --multirun \
    +experiment=$EXPERIMENT \
    model=$MODEL \
    dataset=$DATASET
