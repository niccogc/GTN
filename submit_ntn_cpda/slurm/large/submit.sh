#!/bin/bash
# Submit all large NTN jobs to SLURM
# 16 jobs

# Record submission timestamp
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

mkdir -p logs

sbatch job_ntn_cpda_abalone.sh
sbatch job_ntn_cpda_typei_abalone.sh
sbatch job_ntn_cpda_adult.sh
sbatch job_ntn_cpda_typei_adult.sh
sbatch job_ntn_cpda_ai4i.sh
sbatch job_ntn_cpda_typei_ai4i.sh
sbatch job_ntn_cpda_appliances.sh
sbatch job_ntn_cpda_typei_appliances.sh
sbatch job_ntn_cpda_bank.sh
sbatch job_ntn_cpda_typei_bank.sh
sbatch job_ntn_cpda_mushrooms.sh
sbatch job_ntn_cpda_typei_mushrooms.sh
sbatch job_ntn_cpda_popularity.sh
sbatch job_ntn_cpda_typei_popularity.sh
sbatch job_ntn_cpda_student_dropout.sh
sbatch job_ntn_cpda_typei_student_dropout.sh

# Mark as submitted
echo "Submitted at $TIMESTAMP" > submitted_submit_$TIMESTAMP
echo "Submitted 16 jobs at $(date)"
