#!/bin/bash
# Submit all medium GTN array jobs to SLURM
# 3 jobs

# Record submission timestamp
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

mkdir -p logs

sbatch array_medium_01.sh
sbatch array_medium_02.sh
sbatch array_medium_03.sh

# Mark as submitted
echo "Submitted at $TIMESTAMP" > submitted_submit_$TIMESTAMP
echo "Submitted 3 jobs at $(date)"
