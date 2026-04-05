#!/bin/sh
# Submit all small NTN jobs to HPC
# 8 jobs

# Record submission timestamp
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

mkdir -p logs

bsub < job_ntn_tnml_p_breast.sh
bsub < job_ntn_tnml_f_breast.sh
bsub < job_ntn_tnml_p_hearth.sh
bsub < job_ntn_tnml_f_hearth.sh
bsub < job_ntn_tnml_p_iris.sh
bsub < job_ntn_tnml_f_iris.sh
bsub < job_ntn_tnml_p_wine.sh
bsub < job_ntn_tnml_f_wine.sh

# Mark as submitted
echo "Submitted at $TIMESTAMP" > submitted_submit_$TIMESTAMP
echo "Submitted 8 jobs at $(date)"
