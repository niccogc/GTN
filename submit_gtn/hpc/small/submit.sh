#!/bin/sh
# Submit all small GTN jobs to HPC
# 24 jobs

# Record submission timestamp
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

mkdir -p logs

bsub < job_gtn_mpo2_breast.sh
bsub < job_gtn_lmpo2_breast.sh
bsub < job_gtn_mmpo2_breast.sh
bsub < job_gtn_mpo2_typei_breast.sh
bsub < job_gtn_lmpo2_typei_breast.sh
bsub < job_gtn_mmpo2_typei_breast.sh
bsub < job_gtn_mpo2_hearth.sh
bsub < job_gtn_lmpo2_hearth.sh
bsub < job_gtn_mmpo2_hearth.sh
bsub < job_gtn_mpo2_typei_hearth.sh
bsub < job_gtn_lmpo2_typei_hearth.sh
bsub < job_gtn_mmpo2_typei_hearth.sh
bsub < job_gtn_mpo2_iris.sh
bsub < job_gtn_lmpo2_iris.sh
bsub < job_gtn_mmpo2_iris.sh
bsub < job_gtn_mpo2_typei_iris.sh
bsub < job_gtn_lmpo2_typei_iris.sh
bsub < job_gtn_mmpo2_typei_iris.sh
bsub < job_gtn_mpo2_wine.sh
bsub < job_gtn_lmpo2_wine.sh
bsub < job_gtn_mmpo2_wine.sh
bsub < job_gtn_mpo2_typei_wine.sh
bsub < job_gtn_lmpo2_typei_wine.sh
bsub < job_gtn_mmpo2_typei_wine.sh

# Mark as submitted
echo "Submitted at $TIMESTAMP" > submitted_submit_$TIMESTAMP
echo "Submitted 24 jobs at $(date)"
