#!/bin/bash
# Submit small dataset NTN jobs to SLURM
# 8 jobs

# Record submission timestamp
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

mkdir -p logs

sbatch slurm/small/job_ntn_tnml_f_breast.sh
sbatch slurm/small/job_ntn_tnml_p_breast.sh
sbatch slurm/small/job_ntn_tnml_f_hearth.sh
sbatch slurm/small/job_ntn_tnml_p_hearth.sh
sbatch slurm/small/job_ntn_tnml_f_iris.sh
sbatch slurm/small/job_ntn_tnml_p_iris.sh
sbatch slurm/small/job_ntn_tnml_f_wine.sh
sbatch slurm/small/job_ntn_tnml_p_wine.sh

# Mark as submitted
echo "Submitted at $TIMESTAMP" > submitted_submit_small_slurm_$TIMESTAMP
echo "Submitted 8 jobs at $(date)"
