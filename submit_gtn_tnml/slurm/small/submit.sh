#!/bin/bash
# Submit all small GTN jobs to SLURM
# 8 jobs

# Record submission timestamp
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

mkdir -p logs

sbatch job_gtn_tnml_f_breast.sh
sbatch job_gtn_tnml_p_breast.sh
sbatch job_gtn_tnml_f_hearth.sh
sbatch job_gtn_tnml_p_hearth.sh
sbatch job_gtn_tnml_f_iris.sh
sbatch job_gtn_tnml_p_iris.sh
sbatch job_gtn_tnml_f_wine.sh
sbatch job_gtn_tnml_p_wine.sh

# Mark as submitted
echo "Submitted at $TIMESTAMP" > submitted_submit_$TIMESTAMP
echo "Submitted 8 jobs at $(date)"
