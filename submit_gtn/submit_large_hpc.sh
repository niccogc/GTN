#!/bin/sh
# Submit large dataset GTN jobs to HPC
# 48 jobs

# Record submission timestamp
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

mkdir -p logs

bsub < hpc/large/job_gtn_mpo2_abalone.sh
bsub < hpc/large/job_gtn_lmpo2_abalone.sh
bsub < hpc/large/job_gtn_mmpo2_abalone.sh
bsub < hpc/large/job_gtn_mpo2_typei_abalone.sh
bsub < hpc/large/job_gtn_lmpo2_typei_abalone.sh
bsub < hpc/large/job_gtn_mmpo2_typei_abalone.sh
bsub < hpc/large/job_gtn_mpo2_adult.sh
bsub < hpc/large/job_gtn_lmpo2_adult.sh
bsub < hpc/large/job_gtn_mmpo2_adult.sh
bsub < hpc/large/job_gtn_mpo2_typei_adult.sh
bsub < hpc/large/job_gtn_lmpo2_typei_adult.sh
bsub < hpc/large/job_gtn_mmpo2_typei_adult.sh
bsub < hpc/large/job_gtn_mpo2_ai4i.sh
bsub < hpc/large/job_gtn_lmpo2_ai4i.sh
bsub < hpc/large/job_gtn_mmpo2_ai4i.sh
bsub < hpc/large/job_gtn_mpo2_typei_ai4i.sh
bsub < hpc/large/job_gtn_lmpo2_typei_ai4i.sh
bsub < hpc/large/job_gtn_mmpo2_typei_ai4i.sh
bsub < hpc/large/job_gtn_mpo2_appliances.sh
bsub < hpc/large/job_gtn_lmpo2_appliances.sh
bsub < hpc/large/job_gtn_mmpo2_appliances.sh
bsub < hpc/large/job_gtn_mpo2_typei_appliances.sh
bsub < hpc/large/job_gtn_lmpo2_typei_appliances.sh
bsub < hpc/large/job_gtn_mmpo2_typei_appliances.sh
bsub < hpc/large/job_gtn_mpo2_bank.sh
bsub < hpc/large/job_gtn_lmpo2_bank.sh
bsub < hpc/large/job_gtn_mmpo2_bank.sh
bsub < hpc/large/job_gtn_mpo2_typei_bank.sh
bsub < hpc/large/job_gtn_lmpo2_typei_bank.sh
bsub < hpc/large/job_gtn_mmpo2_typei_bank.sh
bsub < hpc/large/job_gtn_mpo2_mushrooms.sh
bsub < hpc/large/job_gtn_lmpo2_mushrooms.sh
bsub < hpc/large/job_gtn_mmpo2_mushrooms.sh
bsub < hpc/large/job_gtn_mpo2_typei_mushrooms.sh
bsub < hpc/large/job_gtn_lmpo2_typei_mushrooms.sh
bsub < hpc/large/job_gtn_mmpo2_typei_mushrooms.sh
bsub < hpc/large/job_gtn_mpo2_popularity.sh
bsub < hpc/large/job_gtn_lmpo2_popularity.sh
bsub < hpc/large/job_gtn_mmpo2_popularity.sh
bsub < hpc/large/job_gtn_mpo2_typei_popularity.sh
bsub < hpc/large/job_gtn_lmpo2_typei_popularity.sh
bsub < hpc/large/job_gtn_mmpo2_typei_popularity.sh
bsub < hpc/large/job_gtn_mpo2_student_dropout.sh
bsub < hpc/large/job_gtn_lmpo2_student_dropout.sh
bsub < hpc/large/job_gtn_mmpo2_student_dropout.sh
bsub < hpc/large/job_gtn_mpo2_typei_student_dropout.sh
bsub < hpc/large/job_gtn_lmpo2_typei_student_dropout.sh
bsub < hpc/large/job_gtn_mmpo2_typei_student_dropout.sh

# Mark as submitted
echo "Submitted at $TIMESTAMP" > submitted_submit_large_hpc_$TIMESTAMP
echo "Submitted 48 jobs at $(date)"
