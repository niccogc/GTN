#!/bin/bash
# Submit large dataset NTN jobs to SLURM
# 16 jobs

# Record submission timestamp
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

mkdir -p logs

sbatch slurm/large/job_ntn_tnml_f_abalone.sh
sbatch slurm/large/job_ntn_tnml_p_abalone.sh
sbatch slurm/large/job_ntn_tnml_f_adult.sh
sbatch slurm/large/job_ntn_tnml_p_adult.sh
sbatch slurm/large/job_ntn_tnml_f_ai4i.sh
sbatch slurm/large/job_ntn_tnml_p_ai4i.sh
sbatch slurm/large/job_ntn_tnml_f_appliances.sh
sbatch slurm/large/job_ntn_tnml_p_appliances.sh
sbatch slurm/large/job_ntn_tnml_f_bank.sh
sbatch slurm/large/job_ntn_tnml_p_bank.sh
sbatch slurm/large/job_ntn_tnml_f_mushrooms.sh
sbatch slurm/large/job_ntn_tnml_p_mushrooms.sh
sbatch slurm/large/job_ntn_tnml_f_popularity.sh
sbatch slurm/large/job_ntn_tnml_p_popularity.sh
sbatch slurm/large/job_ntn_tnml_f_student_dropout.sh
sbatch slurm/large/job_ntn_tnml_p_student_dropout.sh

# Mark as submitted
echo "Submitted at $TIMESTAMP" > submitted_submit_large_slurm_$TIMESTAMP
echo "Submitted 16 jobs at $(date)"
