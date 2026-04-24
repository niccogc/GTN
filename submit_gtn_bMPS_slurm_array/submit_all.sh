#!/bin/bash
# Submit ALL GTN array jobs to SLURM
# 5 array scripts

# Record submission timestamp
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

mkdir -p small/logs medium/logs large/logs

sbatch small/array_small_01.sh
sbatch medium/array_medium_01.sh
sbatch medium/array_medium_02.sh
sbatch large/array_large_01.sh
sbatch large/array_large_02.sh

# Mark as submitted
echo "Submitted at $TIMESTAMP" > submitted_submit_all_$TIMESTAMP
echo "Submitted 5 array jobs at $(date)"
