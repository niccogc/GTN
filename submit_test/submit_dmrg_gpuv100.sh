#!/bin/bash
#BSUB -q gpuv100
#BSUB -J "ntn_test[1-20]%20"
#BSUB -W 8:00
#BSUB -n 6
#BSUB -R "rusage[mem=1GB]"
#BSUB -R "span[hosts=1]"
#BSUB -o logs/%J_%I_TEST.out
#BSUB -e logs/%J_%I_TEST.err
#BSUB -gpu "num=1:mode=exclusive_process"

DATASETS=( "iris" "hearth" "winequalityc" "wine" "realstate" "energy_efficiency" "concrete" "abalone"  "ai4i" )

NUM_MODELS=1
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
CURRENT_MODEL=tnml_f
EXPERIMENT=test_dmrg

echo "Task $LSB_JOBINDEX: Dataset=$CURRENT_DATASET, Model=$CURRENT_MODEL, Experiment=$EXPERIMENT"

export OMP_NUM_THREADS=6
export MKL_NUM_THREADS=6
export OPENBLAS_NUM_THREADS=6
export NUMEXPR_NUM_THREADS=6

python run.py --multirun \
    +experiment=$EXPERIMENT \
    model=$CURRENT_MODEL \
    dataset=$CURRENT_DATASET
