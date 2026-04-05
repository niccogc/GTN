#!/bin/bash
# Submit medium dataset GTN array jobs to SLURM
# 3 jobs

# Record submission timestamp
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

mkdir -p logs

sbatch medium/array_medium_01.sh
sbatch medium/array_medium_02.sh
sbatch medium/array_medium_03.sh

# Mark as submitted
echo "Submitted at $TIMESTAMP" > submitted_submit_medium_$TIMESTAMP
echo "Submitted 3 jobs at $(date)"
