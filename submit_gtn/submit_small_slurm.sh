#!/bin/bash
# Submit small dataset GTN jobs to SLURM
# 24 jobs

# Record submission timestamp
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

mkdir -p logs

sbatch slurm/small/job_gtn_mpo2_breast.sh
sbatch slurm/small/job_gtn_lmpo2_breast.sh
sbatch slurm/small/job_gtn_mmpo2_breast.sh
sbatch slurm/small/job_gtn_mpo2_typei_breast.sh
sbatch slurm/small/job_gtn_lmpo2_typei_breast.sh
sbatch slurm/small/job_gtn_mmpo2_typei_breast.sh
sbatch slurm/small/job_gtn_mpo2_hearth.sh
sbatch slurm/small/job_gtn_lmpo2_hearth.sh
sbatch slurm/small/job_gtn_mmpo2_hearth.sh
sbatch slurm/small/job_gtn_mpo2_typei_hearth.sh
sbatch slurm/small/job_gtn_lmpo2_typei_hearth.sh
sbatch slurm/small/job_gtn_mmpo2_typei_hearth.sh
sbatch slurm/small/job_gtn_mpo2_iris.sh
sbatch slurm/small/job_gtn_lmpo2_iris.sh
sbatch slurm/small/job_gtn_mmpo2_iris.sh
sbatch slurm/small/job_gtn_mpo2_typei_iris.sh
sbatch slurm/small/job_gtn_lmpo2_typei_iris.sh
sbatch slurm/small/job_gtn_mmpo2_typei_iris.sh
sbatch slurm/small/job_gtn_mpo2_wine.sh
sbatch slurm/small/job_gtn_lmpo2_wine.sh
sbatch slurm/small/job_gtn_mmpo2_wine.sh
sbatch slurm/small/job_gtn_mpo2_typei_wine.sh
sbatch slurm/small/job_gtn_lmpo2_typei_wine.sh
sbatch slurm/small/job_gtn_mmpo2_typei_wine.sh

# Mark as submitted
echo "Submitted at $TIMESTAMP" > submitted_submit_small_slurm_$TIMESTAMP
echo "Submitted 24 jobs at $(date)"
