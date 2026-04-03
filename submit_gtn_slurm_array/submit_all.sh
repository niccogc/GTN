#!/bin/bash
# Submit ALL GTN array jobs to SLURM
# 21 array scripts

# Record submission timestamp
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

mkdir -p small/logs medium/logs large/logs

sbatch small/array_small_01.sh
sbatch small/array_small_02.sh
sbatch small/array_small_03.sh
sbatch small/array_small_04.sh
sbatch medium/array_medium_01.sh
sbatch medium/array_medium_02.sh
sbatch medium/array_medium_03.sh
sbatch medium/array_medium_04.sh
sbatch medium/array_medium_05.sh
sbatch medium/array_medium_06.sh
sbatch medium/array_medium_07.sh
sbatch medium/array_medium_08.sh
sbatch medium/array_medium_09.sh
sbatch large/array_large_01.sh
sbatch large/array_large_02.sh
sbatch large/array_large_03.sh
sbatch large/array_large_04.sh
sbatch large/array_large_05.sh
sbatch large/array_large_06.sh
sbatch large/array_large_07.sh
sbatch large/array_large_08.sh

# Mark as submitted
echo "Submitted at $TIMESTAMP" > submitted_submit_all_$TIMESTAMP
echo "Submitted 21 array jobs at $(date)"
