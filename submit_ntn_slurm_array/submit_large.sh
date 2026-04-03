#!/bin/bash
# Submit large dataset NTN array jobs to SLURM
# 8 jobs

# Record submission timestamp
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

mkdir -p logs

sbatch large/array_large_01.sh
sbatch large/array_large_02.sh
sbatch large/array_large_03.sh
sbatch large/array_large_04.sh
sbatch large/array_large_05.sh
sbatch large/array_large_06.sh
sbatch large/array_large_07.sh
sbatch large/array_large_08.sh

# Mark as submitted
echo "Submitted at $TIMESTAMP" > submitted_submit_large_$TIMESTAMP
echo "Submitted 8 jobs at $(date)"
