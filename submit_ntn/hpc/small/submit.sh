#!/bin/sh
# Submit all small NTN jobs to HPC
# 40 jobs

# Record submission timestamp
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

mkdir -p logs

bsub < job_ntn_mpo2_breast.sh
bsub < job_ntn_lmpo2_breast.sh
bsub < job_ntn_mmpo2_breast.sh
bsub < job_ntn_mpo2_typei_breast.sh
bsub < job_ntn_lmpo2_typei_breast.sh
bsub < job_ntn_mmpo2_typei_breast.sh
bsub < job_ntn_tnml_p_breast.sh
bsub < job_ntn_tnml_f_breast.sh
bsub < job_ntn_cpda_breast.sh
bsub < job_ntn_cpda_typei_breast.sh
bsub < job_ntn_mpo2_hearth.sh
bsub < job_ntn_lmpo2_hearth.sh
bsub < job_ntn_mmpo2_hearth.sh
bsub < job_ntn_mpo2_typei_hearth.sh
bsub < job_ntn_lmpo2_typei_hearth.sh
bsub < job_ntn_mmpo2_typei_hearth.sh
bsub < job_ntn_tnml_p_hearth.sh
bsub < job_ntn_tnml_f_hearth.sh
bsub < job_ntn_cpda_hearth.sh
bsub < job_ntn_cpda_typei_hearth.sh
bsub < job_ntn_mpo2_iris.sh
bsub < job_ntn_lmpo2_iris.sh
bsub < job_ntn_mmpo2_iris.sh
bsub < job_ntn_mpo2_typei_iris.sh
bsub < job_ntn_lmpo2_typei_iris.sh
bsub < job_ntn_mmpo2_typei_iris.sh
bsub < job_ntn_tnml_p_iris.sh
bsub < job_ntn_tnml_f_iris.sh
bsub < job_ntn_cpda_iris.sh
bsub < job_ntn_cpda_typei_iris.sh
bsub < job_ntn_mpo2_wine.sh
bsub < job_ntn_lmpo2_wine.sh
bsub < job_ntn_mmpo2_wine.sh
bsub < job_ntn_mpo2_typei_wine.sh
bsub < job_ntn_lmpo2_typei_wine.sh
bsub < job_ntn_mmpo2_typei_wine.sh
bsub < job_ntn_tnml_p_wine.sh
bsub < job_ntn_tnml_f_wine.sh
bsub < job_ntn_cpda_wine.sh
bsub < job_ntn_cpda_typei_wine.sh

# Mark as submitted
echo "Submitted at $TIMESTAMP" > submitted_submit_$TIMESTAMP
echo "Submitted 40 jobs at $(date)"
