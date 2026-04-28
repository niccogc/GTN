#!/bin/sh
# Submit all small NTN jobs to HPC
# 8 jobs

# Record submission timestamp
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

mkdir -p logs

bsub < job_ntn_cpda_breast.sh
bsub < job_ntn_cpda_typei_breast.sh
bsub < job_ntn_cpda_hearth.sh
bsub < job_ntn_cpda_typei_hearth.sh
bsub < job_ntn_cpda_iris.sh
bsub < job_ntn_cpda_typei_iris.sh
bsub < job_ntn_cpda_wine.sh
bsub < job_ntn_cpda_typei_wine.sh

# Mark as submitted
echo "Submitted at $TIMESTAMP" > submitted_submit_$TIMESTAMP
echo "Submitted 8 jobs at $(date)"
