#!/bin/bash
# Submit small dataset GTN array jobs to SLURM
# 4 jobs

# Record submission timestamp
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

mkdir -p logs

sbatch small/array_small_01.sh
sbatch small/array_small_02.sh
sbatch small/array_small_03.sh
sbatch small/array_small_04.sh

# Mark as submitted
echo "Submitted at $TIMESTAMP" > submitted_submit_small_$TIMESTAMP
echo "Submitted 4 jobs at $(date)"
