#!/bin/bash
#SBATCH --job-name=ntn-large_02
#SBATCH --output=logs/ntn-large_02_%A_%a.out
#SBATCH --error=logs/ntn-large_02_%A_%a.err
#SBATCH --partition=titans
#SBATCH --time=24:00:00
#SBATCH --mem=2gb
#SBATCH --gres=gpu:Ampere:1
#SBATCH --cpus-per-task=1
#SBATCH --mail-user=nicci@dtu.dk
#SBATCH --mail-type=FAIL,ARRAY_TASKS
#SBATCH --array=1-6

echo "Node: $(hostname)"
echo "Array Job ID: $SLURM_ARRAY_JOB_ID, Task ID: $SLURM_ARRAY_TASK_ID"
echo "Start: $(date +%F-%R:%S)"

export HOME=/home/nicci
cd $HOME/GTN
source $(conda info --base)/etc/profile.d/conda.sh
conda activate gtn

set -a && source $HOME/aim && set +a

# Read parameters from params file (SLURM_SUBMIT_DIR is where sbatch was run from)
PARAMS_FILE="$SLURM_SUBMIT_DIR/params_large_02.txt"
MODEL=$(awk -v TaskID=$SLURM_ARRAY_TASK_ID '$1==TaskID {print $2}' $PARAMS_FILE)
DATASET=$(awk -v TaskID=$SLURM_ARRAY_TASK_ID '$1==TaskID {print $3}' $PARAMS_FILE)

echo "Running: model=$MODEL dataset=$DATASET"

python run.py --multirun +experiment=cpda_gtn_sweep model=$MODEL dataset=$DATASET

echo "Done: $(date +%F-%R:%S)"
