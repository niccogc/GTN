#!/bin/bash
# Submit small dataset NTN jobs to SLURM
# 40 jobs

# Record submission timestamp
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

mkdir -p logs

sbatch slurm/small/job_ntn_mpo2_breast.sh
sbatch slurm/small/job_ntn_lmpo2_breast.sh
sbatch slurm/small/job_ntn_mmpo2_breast.sh
sbatch slurm/small/job_ntn_mpo2_typei_breast.sh
sbatch slurm/small/job_ntn_lmpo2_typei_breast.sh
sbatch slurm/small/job_ntn_mmpo2_typei_breast.sh
sbatch slurm/small/job_ntn_tnml_p_breast.sh
sbatch slurm/small/job_ntn_tnml_f_breast.sh
sbatch slurm/small/job_ntn_cpda_breast.sh
sbatch slurm/small/job_ntn_cpda_typei_breast.sh
sbatch slurm/small/job_ntn_mpo2_hearth.sh
sbatch slurm/small/job_ntn_lmpo2_hearth.sh
sbatch slurm/small/job_ntn_mmpo2_hearth.sh
sbatch slurm/small/job_ntn_mpo2_typei_hearth.sh
sbatch slurm/small/job_ntn_lmpo2_typei_hearth.sh
sbatch slurm/small/job_ntn_mmpo2_typei_hearth.sh
sbatch slurm/small/job_ntn_tnml_p_hearth.sh
sbatch slurm/small/job_ntn_tnml_f_hearth.sh
sbatch slurm/small/job_ntn_cpda_hearth.sh
sbatch slurm/small/job_ntn_cpda_typei_hearth.sh
sbatch slurm/small/job_ntn_mpo2_iris.sh
sbatch slurm/small/job_ntn_lmpo2_iris.sh
sbatch slurm/small/job_ntn_mmpo2_iris.sh
sbatch slurm/small/job_ntn_mpo2_typei_iris.sh
sbatch slurm/small/job_ntn_lmpo2_typei_iris.sh
sbatch slurm/small/job_ntn_mmpo2_typei_iris.sh
sbatch slurm/small/job_ntn_tnml_p_iris.sh
sbatch slurm/small/job_ntn_tnml_f_iris.sh
sbatch slurm/small/job_ntn_cpda_iris.sh
sbatch slurm/small/job_ntn_cpda_typei_iris.sh
sbatch slurm/small/job_ntn_mpo2_wine.sh
sbatch slurm/small/job_ntn_lmpo2_wine.sh
sbatch slurm/small/job_ntn_mmpo2_wine.sh
sbatch slurm/small/job_ntn_mpo2_typei_wine.sh
sbatch slurm/small/job_ntn_lmpo2_typei_wine.sh
sbatch slurm/small/job_ntn_mmpo2_typei_wine.sh
sbatch slurm/small/job_ntn_tnml_p_wine.sh
sbatch slurm/small/job_ntn_tnml_f_wine.sh
sbatch slurm/small/job_ntn_cpda_wine.sh
sbatch slurm/small/job_ntn_cpda_typei_wine.sh

# Mark as submitted
echo "Submitted at $TIMESTAMP" > submitted_submit_small_slurm_$TIMESTAMP
echo "Submitted 40 jobs at $(date)"
