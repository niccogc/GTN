#!/bin/bash
# Submit all small NTN jobs to SLURM
# 8 jobs

# Record submission timestamp
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

mkdir -p logs

sbatch job_ntn_cpda_breast.sh
sbatch job_ntn_cpda_typei_breast.sh
sbatch job_ntn_cpda_hearth.sh
sbatch job_ntn_cpda_typei_hearth.sh
sbatch job_ntn_cpda_iris.sh
sbatch job_ntn_cpda_typei_iris.sh
sbatch job_ntn_cpda_wine.sh
sbatch job_ntn_cpda_typei_wine.sh

# Mark as submitted
echo "Submitted at $TIMESTAMP" > submitted_submit_$TIMESTAMP
echo "Submitted 8 jobs at $(date)"
