#!/bin/bash
#BSUB -q hpc
#BSUB -J "gtn_boson[1-4]%4"
#BSUB -W 2:00
#BSUB -n 4
#BSUB -R "rusage[mem=1GB]"
#BSUB -R "span[hosts=1]"
#BSUB -o logs/%J_%I.out
#BSUB -e logs/%J_%I.err

MODELS=("bosonmps")

DATASETS=(
    "adult" "appliances" "bank" "popularity" )

NUM_MODELS=${#MODELS[@]}
NUM_DATASETS=${#DATASETS[@]}
NUM_EXPERIMENTS=$((NUM_DATASETS * NUM_MODELS))

if [ "$1" == "test" ]; then
    echo "SIMULATING ARRAY:"
    echo "-----------------"

    for ((i=1; i<=NUM_EXPERIMENTS; i++)); do
        IDV=$((i - 1))
        MODEL_IDX=$((IDV % NUM_MODELS))
        DATASET_IDX=$((IDV / NUM_MODELS))

        echo "Index $i: Dataset=${DATASETS[$DATASET_IDX]}, Model=${MODELS[$MODEL_IDX]}"
    done
    exit 0
fi

export HOME=/zhome/6b/e/212868
cd "$HOME/GTN"
source .venv/bin/activate

mkdir -p logs

IDV=$((LSB_JOBINDEX - 1))
MODEL_IDX=$((IDV % NUM_MODELS))
DATASET_IDX=$((IDV / NUM_MODELS))

CURRENT_DATASET=${DATASETS[$DATASET_IDX]}
CURRENT_MODEL=${MODELS[$MODEL_IDX]}
EXPERIMENT="uci_gtn_sweep"

echo "Task $LSB_JOBINDEX: Dataset=$CURRENT_DATASET, Model=$CURRENT_MODEL, Experiment=$EXPERIMENT"

export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4

python run.py --multirun \
    +experiment=$EXPERIMENT \
    model=$CURRENT_MODEL \
    dataset=$CURRENT_DATASET
