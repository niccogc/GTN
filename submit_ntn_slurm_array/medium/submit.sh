#!/bin/bash
# Submit all medium NTN array jobs to SLURM
# 9 jobs

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

# Mark as submitted
echo "Submitted at $TIMESTAMP" > submitted_submit_$TIMESTAMP
echo "Submitted 9 jobs at $(date)"
