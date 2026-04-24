#!/bin/bash
# Submit large dataset GTN array jobs to SLURM
# 2 jobs

# Record submission timestamp
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

mkdir -p logs

sbatch large/array_large_01.sh
sbatch large/array_large_02.sh

# Mark as submitted
echo "Submitted at $TIMESTAMP" > submitted_submit_large_$TIMESTAMP
echo "Submitted 2 jobs at $(date)"
