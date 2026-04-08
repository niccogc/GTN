#!/bin/bash
# Submit small dataset NTN array jobs to SLURM
# 2 jobs

# Record submission timestamp
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

mkdir -p logs

sbatch small/array_small_01.sh
sbatch small/array_small_02.sh

# Mark as submitted
echo "Submitted at $TIMESTAMP" > submitted_submit_small_$TIMESTAMP
echo "Submitted 2 jobs at $(date)"
