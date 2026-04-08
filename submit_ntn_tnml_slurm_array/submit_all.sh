#!/bin/bash
# Submit ALL NTN array jobs to SLURM
# 8 array scripts

# Record submission timestamp
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

mkdir -p small/logs medium/logs large/logs

sbatch small/array_small_01.sh
sbatch small/array_small_02.sh
sbatch medium/array_medium_01.sh
sbatch medium/array_medium_02.sh
sbatch medium/array_medium_03.sh
sbatch large/array_large_01.sh
sbatch large/array_large_02.sh
sbatch large/array_large_03.sh

# Mark as submitted
echo "Submitted at $TIMESTAMP" > submitted_submit_all_$TIMESTAMP
echo "Submitted 8 array jobs at $(date)"
