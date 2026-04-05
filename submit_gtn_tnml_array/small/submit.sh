#!/bin/bash
# Submit all small GTN array jobs to SLURM
# 2 jobs

# Record submission timestamp
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

mkdir -p logs

sbatch array_small_01.sh
sbatch array_small_02.sh

# Mark as submitted
echo "Submitted at $TIMESTAMP" > submitted_submit_$TIMESTAMP
echo "Submitted 2 jobs at $(date)"
