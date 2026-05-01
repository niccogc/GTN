#!/bin/bash
#BSUB -q hpc
#BSUB -J "ntn_grid_full[1-150]%20"
#BSUB -W 8:00
#BSUB -n 4
#BSUB -R "rusage[mem=750MB]"
#BSUB -R "span[hosts=1]"
#BSUB -o logs/%J_%I.out
#BSUB -e logs/%J_%I.err

MODELS=("cpda" "cpda_typei" "lmpo2" "lmpo2_typei" "mpo2" "mpo2_typei" "mmpo2" "mmpo2_typei" "tnml_f" "tnml_p")

DATASETS=(
    "abalone" "adult" "ai4i" "appliances" "bank" "bike" "breast"
    "car_evaluation" "concrete" "energy_efficiency" "hearth"
    "iris" "mushrooms" "obesity" "popularity" "realstate"
    "seoulBike" "student_dropout" "student_perf" "wine"
    "winequalityc"
)

COMPLETED_DATASETS=(
    "abalone" "bike" "concrete" "energy_efficiency"
    "iris" "realstate" )

# Filter datasets
FILTERED_DATASETS=()
for ds in "${DATASETS[@]}"; do
    skip=false
    for completed in "${COMPLETED_DATASETS[@]}"; do
        if [[ "$ds" == "$completed" ]]; then
            skip=true
            break
        fi
    done

    if ! $skip; then
        FILTERED_DATASETS+=("$ds")
    fi
done

DATASETS=("${FILTERED_DATASETS[@]}")

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

if [[ "$CURRENT_MODEL" == "cpda" || "$CURRENT_MODEL" == "cpda_typei" ]]; then
    EXPERIMENT="cpda_ntn_sweep"
else
    EXPERIMENT="uci_ntn_sweep"
fi

echo "Task $LSB_JOBINDEX: Dataset=$CURRENT_DATASET, Model=$CURRENT_MODEL, Experiment=$EXPERIMENT"

export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4

python run.py --multirun \
    +experiment=$EXPERIMENT \
    model=$CURRENT_MODEL \
    dataset=$CURRENT_DATASET
