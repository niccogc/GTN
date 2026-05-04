#!/bin/bash
#BSUB -q h100
#BSUB -J "ntn_missing[1-20]%20"
#BSUB -W 8:00
#BSUB -n 4
#BSUB -R "rusage[mem=500MB]"
#BSUB -R "span[hosts=1]"
#BSUB -o logs/%J_%I.out
#BSUB -e logs/%J_%I.err

# Missing model-dataset combinations from README.md (NTN only, ignoring GTN)
# Total: 20 combinations

declare -a COMBINATIONS=(
    # LMPO2TypeI (11 datasets)
    "lmpo2_typei breast"
    "lmpo2_typei car_evaluation"
    "lmpo2_typei obesity"
    "lmpo2_typei seoulBike"
    "lmpo2_typei student_perf"
    "lmpo2_typei adult"
    "lmpo2_typei appliances"
    "lmpo2_typei bank"
    "lmpo2_typei mushrooms"
    "lmpo2_typei popularity"
    "lmpo2_typei student_dropout"
    # MMPO2 (3 datasets)
    "mmpo2 adult"
    "mmpo2 mushrooms"
    "mmpo2 popularity"
    # MMPO2TypeI (4 datasets)
    "mmpo2_typei adult"
    "mmpo2_typei bank"
    "mmpo2_typei mushrooms"
    "mmpo2_typei popularity"
    # TNML_F (2 datasets)
    "tnml_f adult"
    "tnml_f bank"
)

NUM_COMBINATIONS=${#COMBINATIONS[@]}

if [ "$1" == "test" ]; then
    echo "SIMULATING ARRAY:"
    echo "-----------------"
    echo "Total combinations: $NUM_COMBINATIONS"
    echo ""

    for ((i=1; i<=NUM_COMBINATIONS; i++)); do
        IDX=$((i - 1))
        read -r MODEL DATASET <<< "${COMBINATIONS[$IDX]}"
        echo "Index $i: Model=$MODEL, Dataset=$DATASET"
    done
    exit 0
fi

mkdir -p logs

IDX=$((LSB_JOBINDEX - 1))
read -r CURRENT_MODEL CURRENT_DATASET <<< "${COMBINATIONS[$IDX]}"

echo "Task $LSB_JOBINDEX: Model=$CURRENT_MODEL, Dataset=$CURRENT_DATASET"

python run.py --multirun \
    +experiment=uci_ntn_sweep \
    model=$CURRENT_MODEL \
    dataset=$CURRENT_DATASET
