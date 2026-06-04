#!/bin/bash
#SBATCH --job-name=dmrg_sweep
#SBATCH --output=/home/nicci/GTN/logs/gtn_%A_%a.out
#SBATCH --error=/home/nicci/GTN/logs/gtn_%A_%a.err
#SBATCH --partition=titans
#SBATCH --time=24:00:00
#SBATCH --mem=4gb
#SBATCH --gres=gpu
#SBATCH --cpus-per-task=8
#SBATCH --array=1-20%10

export HOME=/home/nicci
cd $HOME/GTN
source $(conda info --base)/etc/profile.d/conda.sh
conda activate gtn

mkdir -p logs

NUM_EXPERIMENTS=${#COMBINATIONS_DMRG[@]}

IDV=$((LSB_JOBINDEX - 1))
read MODEL DATASET <<< "${COMBINATIONS_DMRG[$IDV]}"
echo "DMRG: $MODEL $DATASET"

echo "Task $LSB_JOBINDEX: Dataset=$DATASET, Model=$MODEL, Experiment=$EXPERIMENT"

export OMP_NUM_THREADS=6
export MKL_NUM_THREADS=6
export OPENBLAS_NUM_THREADS=6
export NUMEXPR_NUM_THREADS=6

python run.py --multirun \
    +experiment=dmrg_sweep \
    model=$MODEL \
    dataset=$DATASET
