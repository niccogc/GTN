#!/bin/bash
# Submit all small GTN jobs to SLURM
# 40 jobs

# Record submission timestamp
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

mkdir -p logs

sbatch job_gtn_mpo2_breast.sh
sbatch job_gtn_lmpo2_breast.sh
sbatch job_gtn_mmpo2_breast.sh
sbatch job_gtn_mpo2_typei_breast.sh
sbatch job_gtn_lmpo2_typei_breast.sh
sbatch job_gtn_mmpo2_typei_breast.sh
sbatch job_gtn_tnml_p_breast.sh
sbatch job_gtn_tnml_f_breast.sh
sbatch job_gtn_cpda_breast.sh
sbatch job_gtn_cpda_typei_breast.sh
sbatch job_gtn_mpo2_hearth.sh
sbatch job_gtn_lmpo2_hearth.sh
sbatch job_gtn_mmpo2_hearth.sh
sbatch job_gtn_mpo2_typei_hearth.sh
sbatch job_gtn_lmpo2_typei_hearth.sh
sbatch job_gtn_mmpo2_typei_hearth.sh
sbatch job_gtn_tnml_p_hearth.sh
sbatch job_gtn_tnml_f_hearth.sh
sbatch job_gtn_cpda_hearth.sh
sbatch job_gtn_cpda_typei_hearth.sh
sbatch job_gtn_mpo2_iris.sh
sbatch job_gtn_lmpo2_iris.sh
sbatch job_gtn_mmpo2_iris.sh
sbatch job_gtn_mpo2_typei_iris.sh
sbatch job_gtn_lmpo2_typei_iris.sh
sbatch job_gtn_mmpo2_typei_iris.sh
sbatch job_gtn_tnml_p_iris.sh
sbatch job_gtn_tnml_f_iris.sh
sbatch job_gtn_cpda_iris.sh
sbatch job_gtn_cpda_typei_iris.sh
sbatch job_gtn_mpo2_wine.sh
sbatch job_gtn_lmpo2_wine.sh
sbatch job_gtn_mmpo2_wine.sh
sbatch job_gtn_mpo2_typei_wine.sh
sbatch job_gtn_lmpo2_typei_wine.sh
sbatch job_gtn_mmpo2_typei_wine.sh
sbatch job_gtn_tnml_p_wine.sh
sbatch job_gtn_tnml_f_wine.sh
sbatch job_gtn_cpda_wine.sh
sbatch job_gtn_cpda_typei_wine.sh

# Mark as submitted
echo "Submitted at $TIMESTAMP" > submitted_submit_$TIMESTAMP
echo "Submitted 40 jobs at $(date)"
