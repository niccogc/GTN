#!/bin/bash
# Submit all large GTN jobs to SLURM
# 16 jobs

# Record submission timestamp
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

mkdir -p logs

sbatch job_gtn_tnml_f_abalone.sh
sbatch job_gtn_tnml_p_abalone.sh
sbatch job_gtn_tnml_f_adult.sh
sbatch job_gtn_tnml_p_adult.sh
sbatch job_gtn_tnml_f_ai4i.sh
sbatch job_gtn_tnml_p_ai4i.sh
sbatch job_gtn_tnml_f_appliances.sh
sbatch job_gtn_tnml_p_appliances.sh
sbatch job_gtn_tnml_f_bank.sh
sbatch job_gtn_tnml_p_bank.sh
sbatch job_gtn_tnml_f_mushrooms.sh
sbatch job_gtn_tnml_p_mushrooms.sh
sbatch job_gtn_tnml_f_popularity.sh
sbatch job_gtn_tnml_p_popularity.sh
sbatch job_gtn_tnml_f_student_dropout.sh
sbatch job_gtn_tnml_p_student_dropout.sh

# Mark as submitted
echo "Submitted at $TIMESTAMP" > submitted_submit_$TIMESTAMP
echo "Submitted 16 jobs at $(date)"
