#!/bin/bash
# Submit all large NTN array jobs to SLURM
# 12 jobs

# Record submission timestamp
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

mkdir -p logs

sbatch array_large_01.sh
sbatch array_large_02.sh
sbatch array_large_03.sh
sbatch array_large_04.sh
sbatch array_large_05.sh
sbatch array_large_06.sh
sbatch array_large_07.sh
sbatch array_large_08.sh
sbatch array_large_09.sh
sbatch array_large_10.sh
sbatch array_large_11.sh
sbatch array_large_12.sh

# Mark as submitted
echo "Submitted at $TIMESTAMP" > submitted_submit_$TIMESTAMP
echo "Submitted 12 jobs at $(date)"
