#!/bin/bash
#BSUB -q hpc
#BSUB -J "ntn_grid_full[1-200]%20"
#BSUB -W 12:00
#BSUB -n 4
#BSUB -R "rusage[mem=1GB]"
#BSUB -R "span[hosts=1]"
#BSUB -o logs/%J_%I.out
#BSUB -e logs/%J_%I.err

# --- DRY RUN / TEST LOGIC ---
# Run locally with: bash your_script.sh test
MODELS=("cpda" "cpda_typei" "lmpo2" "lmpo2_typei" "mpo2" "mpo2_typei" "mmpo2" "mmpo2_typei" "tnml_f" "tnml_p")
NUM_MODELS=${#MODELS[@]}
DATASETS=("adult" "ai4i" "appliances" "bank" "bike" "breast" "car_evaluation" "concrete" "energy_efficiency" "hearth" "iris" "mushrooms" "obesity" "popularity" "realstate" "seoulBike" "student_dropout" "student_perf" "wine" "winequalityc"
)

NUM_DATASETS=${#DATASETS[@]}
NUM_EXPERIMENTS=$((NUM_DATASETS*NUM_MODELS))
if [ "$1" == "test" ]; then
    echo "SIMULATING FULL ARRAY MAPPING (1-44):"
    echo "--------------------------------------"
for ((i=1; i<=NUM_EXPERIMENTS; i++)); do
        IDV=$((i - 1))
        M_IDX=$((IDV % NUM_MODELS))
        D_IDX=$((IDV / NUM_MODELS))
        echo "Index $i: Dataset=${DATASETS[$D_IDX]}, Model=${MODELS[$M_IDX]}"
    done
    exit 0
fi
# ----------------------------

export HOME=/zhome/6b/e/212868
cd $HOME/BMPO
source .venv/bin/activate

mkdir -p logs

# LSF logic for the actual cluster run
IDV=$((LSB_JOBINDEX - 1))
MODEL_IDX=$((IDV % NUM_MODELS))
DATASET_IDX=$((IDV / NUM_MODELS))

CURRENT_DATASET=${DATASETS[$DATASET_IDX]}
CURRENT_MODEL=${MODELS[$MODEL_IDX]}
echo "Task $LSB_JOBINDEX: Dataset=$CURRENT_DATASET, Model=$CURRENT_MODEL"

export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

python run.py --multirun +experiment=uci_ntn_sweep model=$CURRENT_MODEL dataset=$CURRENT_DATASET
