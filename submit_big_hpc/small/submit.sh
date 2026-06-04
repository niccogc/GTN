#!/bin/bash
# Submit all small NTN array jobs to SLURM
# 7 jobs

# Record submission timestamp
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

mkdir -p logs

sbatch array_small_01.sh
sbatch array_small_02.sh
sbatch array_small_03.sh
sbatch array_small_04.sh
sbatch array_small_05.sh
sbatch array_small_06.sh
sbatch array_small_07.sh

# Mark as submitted
echo "Submitted at $TIMESTAMP" > submitted_submit_$TIMESTAMP
echo "Submitted 7 jobs at $(date)"
