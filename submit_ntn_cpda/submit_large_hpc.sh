#!/bin/sh
# Submit large dataset NTN jobs to HPC
# 16 jobs

# Record submission timestamp
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

mkdir -p logs

bsub < hpc/large/job_ntn_cpda_abalone.sh
bsub < hpc/large/job_ntn_cpda_typei_abalone.sh
bsub < hpc/large/job_ntn_cpda_adult.sh
bsub < hpc/large/job_ntn_cpda_typei_adult.sh
bsub < hpc/large/job_ntn_cpda_ai4i.sh
bsub < hpc/large/job_ntn_cpda_typei_ai4i.sh
bsub < hpc/large/job_ntn_cpda_appliances.sh
bsub < hpc/large/job_ntn_cpda_typei_appliances.sh
bsub < hpc/large/job_ntn_cpda_bank.sh
bsub < hpc/large/job_ntn_cpda_typei_bank.sh
bsub < hpc/large/job_ntn_cpda_mushrooms.sh
bsub < hpc/large/job_ntn_cpda_typei_mushrooms.sh
bsub < hpc/large/job_ntn_cpda_popularity.sh
bsub < hpc/large/job_ntn_cpda_typei_popularity.sh
bsub < hpc/large/job_ntn_cpda_student_dropout.sh
bsub < hpc/large/job_ntn_cpda_typei_student_dropout.sh

# Mark as submitted
echo "Submitted at $TIMESTAMP" > submitted_submit_large_hpc_$TIMESTAMP
echo "Submitted 16 jobs at $(date)"
