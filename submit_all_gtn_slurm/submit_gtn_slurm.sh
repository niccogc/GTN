#!/bin/bash
#SBATCH --job-name=gtn_sweep
#SBATCH --output=/home/nicci/GTN/logs/gtn_%A_%a.out
#SBATCH --error=/home/nicci/GTN/logs/gtn_%A_%a.err
#SBATCH --partition=titans
#SBATCH --time=1:00:00
#SBATCH --mem=2gb
#SBATCH --gres=gpu
#SBATCH --cpus-per-task=2
#SBATCH --array=1-220%10

MODELS=("cpda" "cpda_typei" "lmpo2" "lmpo2_typei" "mpo2" "mpo2_typei" "mmpo2" "mmpo2_typei" "tnml_f" "tnml_p" "bosonmps")

DATASETS=(
    "abalone" "adult" "ai4i" "bank" "bike" "breast"
    "car_evaluation" "concrete" "energy_efficiency" "hearth"
    "iris" "mushrooms" "obesity" "popularity" "realstate"
    "seoulBike" "student_dropout" "student_perf" "wine"
    "winequalityc"
)

# "appliances" 

NUM_MODELS=${#MODELS[@]}
NUM_DATASETS=${#DATASETS[@]}
NUM_EXPERIMENTS=$((NUM_DATASETS * NUM_MODELS))

export HOME=/home/nicci
cd $HOME/GTN
source $(conda info --base)/etc/profile.d/conda.sh
conda activate gtn

mkdir -p logs

IDV=$((SLURM_ARRAY_TASK_ID - 1))
MODEL_IDX=$((IDV % NUM_MODELS))
DATASET_IDX=$((IDV / NUM_MODELS))

CURRENT_DATASET=${DATASETS[$DATASET_IDX]}
CURRENT_MODEL=${MODELS[$MODEL_IDX]}

if [[ "$CURRENT_MODEL" == "cpda" || "$CURRENT_MODEL" == "cpda_typei" ]]; then
    EXPERIMENT="cpda_gtn_sweep"
else
    EXPERIMENT="uci_gtn_sweep"
fi

echo "Task $SLURM_ARRAY_TASK_ID: Dataset=$CURRENT_DATASET, Model=$CURRENT_MODEL, Experiment=$EXPERIMENT"

export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export NUMEXPR_NUM_THREADS=2

python run.py --multirun \
    +experiment=$EXPERIMENT \
    model=$CURRENT_MODEL \
    dataset=$CURRENT_DATASET
