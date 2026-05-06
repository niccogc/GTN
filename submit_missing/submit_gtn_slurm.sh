#!/bin/bash
#SBATCH --job-name=gtn_sweep
#SBATCH --output=/home/nicci/GTN/logs/gtn_%A_%a.out
#SBATCH --error=/home/nicci/GTN/logs/gtn_%A_%a.err
#SBATCH --partition=titans
#SBATCH --time=2:00:00
#SBATCH --mem=2gb
#SBATCH --gres=gpu
#SBATCH --cpus-per-task=2
#SBATCH --array=1-64%10

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

export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export NUMEXPR_NUM_THREADS=2

python run.py --multirun \
    +experiment=$EXPERIMENT \
    model=$MODEL \
    dataset=$DATASET
