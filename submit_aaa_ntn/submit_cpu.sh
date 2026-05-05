#!/bin/bash
#BSUB -q hpc
#BSUB -J "ntn_grid_full[1-40]%20"
#BSUB -W 4:00
#BSUB -n 6
#BSUB -R "rusage[mem=3GB]"
#BSUB -R "span[hosts=1]"
#BSUB -o logs/%J_%I_cpu_newntn.out
#BSUB -e logs/%J_%I_cpu_newntn.err

MODELS=("cpda" "cpda_typei" "lmpo2" "lmpo2_typei" "mpo2" "mpo2_typei" "mmpo2" "mmpo2_typei" "tnml_f" "tnml_p")

DATASETS=(
    # COMPLETED
    # 
    # "iris"
    # "realstate"
    # "concrete"
    # "energy_efficiency"
    # 
    # "breast"
    # "car_evaluation"
    # "wine"
    #medium
    # "winequalityc"
    # "abalone"
    # "appliances"
    # "bank"
    # "hearth"
    # "student_perf"
    "ai4i"
    "mushrooms"
    "seoulBike"
    "student_dropout"
    # BIG MAYBE?
    # "bike"
    # "obesity"
    # "adult"
    # "popularity"
)

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

cd "~/GTN"
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

export OMP_NUM_THREADS=6
export MKL_NUM_THREADS=6
export OPENBLAS_NUM_THREADS=6
export NUMEXPR_NUM_THREADS=6

python run.py --multirun \
    +experiment=$EXPERIMENT \
    model=$CURRENT_MODEL \
    dataset=$CURRENT_DATASET
