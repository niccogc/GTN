#!/bin/sh
# Submit small dataset NTN jobs to HPC
# 8 jobs

# Record submission timestamp
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

mkdir -p logs

bsub < hpc/small/job_ntn_tnml_f_breast.sh
bsub < hpc/small/job_ntn_tnml_p_breast.sh
bsub < hpc/small/job_ntn_tnml_f_hearth.sh
bsub < hpc/small/job_ntn_tnml_p_hearth.sh
bsub < hpc/small/job_ntn_tnml_f_iris.sh
bsub < hpc/small/job_ntn_tnml_p_iris.sh
bsub < hpc/small/job_ntn_tnml_f_wine.sh
bsub < hpc/small/job_ntn_tnml_p_wine.sh

# Mark as submitted
echo "Submitted at $TIMESTAMP" > submitted_submit_small_hpc_$TIMESTAMP
echo "Submitted 8 jobs at $(date)"
