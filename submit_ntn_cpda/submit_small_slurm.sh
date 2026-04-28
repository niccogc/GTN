#!/bin/bash
# Submit small dataset NTN jobs to SLURM
# 8 jobs

# Record submission timestamp
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

mkdir -p logs

sbatch slurm/small/job_ntn_cpda_breast.sh
sbatch slurm/small/job_ntn_cpda_typei_breast.sh
sbatch slurm/small/job_ntn_cpda_hearth.sh
sbatch slurm/small/job_ntn_cpda_typei_hearth.sh
sbatch slurm/small/job_ntn_cpda_iris.sh
sbatch slurm/small/job_ntn_cpda_typei_iris.sh
sbatch slurm/small/job_ntn_cpda_wine.sh
sbatch slurm/small/job_ntn_cpda_typei_wine.sh

# Mark as submitted
echo "Submitted at $TIMESTAMP" > submitted_submit_small_slurm_$TIMESTAMP
echo "Submitted 8 jobs at $(date)"
