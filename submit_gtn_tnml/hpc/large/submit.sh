#!/bin/sh
# Submit all large GTN jobs to HPC
# 16 jobs

# Record submission timestamp
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

mkdir -p logs

bsub < job_gtn_tnml_p_abalone.sh
bsub < job_gtn_tnml_f_abalone.sh
bsub < job_gtn_tnml_p_adult.sh
bsub < job_gtn_tnml_f_adult.sh
bsub < job_gtn_tnml_p_ai4i.sh
bsub < job_gtn_tnml_f_ai4i.sh
bsub < job_gtn_tnml_p_appliances.sh
bsub < job_gtn_tnml_f_appliances.sh
bsub < job_gtn_tnml_p_bank.sh
bsub < job_gtn_tnml_f_bank.sh
bsub < job_gtn_tnml_p_mushrooms.sh
bsub < job_gtn_tnml_f_mushrooms.sh
bsub < job_gtn_tnml_p_popularity.sh
bsub < job_gtn_tnml_f_popularity.sh
bsub < job_gtn_tnml_p_student_dropout.sh
bsub < job_gtn_tnml_f_student_dropout.sh

# Mark as submitted
echo "Submitted at $TIMESTAMP" > submitted_submit_$TIMESTAMP
echo "Submitted 16 jobs at $(date)"
