#!/bin/bash
# Submit all medium NTN array jobs to SLURM
# 15 jobs

# Record submission timestamp
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

mkdir -p logs

sbatch array_medium_01.sh
sbatch array_medium_02.sh
sbatch array_medium_03.sh
sbatch array_medium_04.sh
sbatch array_medium_05.sh
sbatch array_medium_06.sh
sbatch array_medium_07.sh
sbatch array_medium_08.sh
sbatch array_medium_09.sh
sbatch array_medium_10.sh
sbatch array_medium_11.sh
sbatch array_medium_12.sh
sbatch array_medium_13.sh
sbatch array_medium_14.sh
sbatch array_medium_15.sh

# Mark as submitted
echo "Submitted at $TIMESTAMP" > submitted_submit_$TIMESTAMP
echo "Submitted 15 jobs at $(date)"
