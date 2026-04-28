#!/bin/bash
# Submit all large GTN jobs to SLURM
# 80 jobs

# Record submission timestamp
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

mkdir -p logs

sbatch job_gtn_mpo2_abalone.sh
sbatch job_gtn_lmpo2_abalone.sh
sbatch job_gtn_mmpo2_abalone.sh
sbatch job_gtn_mpo2_typei_abalone.sh
sbatch job_gtn_lmpo2_typei_abalone.sh
sbatch job_gtn_mmpo2_typei_abalone.sh
sbatch job_gtn_tnml_p_abalone.sh
sbatch job_gtn_tnml_f_abalone.sh
sbatch job_gtn_cpda_abalone.sh
sbatch job_gtn_cpda_typei_abalone.sh
sbatch job_gtn_mpo2_adult.sh
sbatch job_gtn_lmpo2_adult.sh
sbatch job_gtn_mmpo2_adult.sh
sbatch job_gtn_mpo2_typei_adult.sh
sbatch job_gtn_lmpo2_typei_adult.sh
sbatch job_gtn_mmpo2_typei_adult.sh
sbatch job_gtn_tnml_p_adult.sh
sbatch job_gtn_tnml_f_adult.sh
sbatch job_gtn_cpda_adult.sh
sbatch job_gtn_cpda_typei_adult.sh
sbatch job_gtn_mpo2_ai4i.sh
sbatch job_gtn_lmpo2_ai4i.sh
sbatch job_gtn_mmpo2_ai4i.sh
sbatch job_gtn_mpo2_typei_ai4i.sh
sbatch job_gtn_lmpo2_typei_ai4i.sh
sbatch job_gtn_mmpo2_typei_ai4i.sh
sbatch job_gtn_tnml_p_ai4i.sh
sbatch job_gtn_tnml_f_ai4i.sh
sbatch job_gtn_cpda_ai4i.sh
sbatch job_gtn_cpda_typei_ai4i.sh
sbatch job_gtn_mpo2_appliances.sh
sbatch job_gtn_lmpo2_appliances.sh
sbatch job_gtn_mmpo2_appliances.sh
sbatch job_gtn_mpo2_typei_appliances.sh
sbatch job_gtn_lmpo2_typei_appliances.sh
sbatch job_gtn_mmpo2_typei_appliances.sh
sbatch job_gtn_tnml_p_appliances.sh
sbatch job_gtn_tnml_f_appliances.sh
sbatch job_gtn_cpda_appliances.sh
sbatch job_gtn_cpda_typei_appliances.sh
sbatch job_gtn_mpo2_bank.sh
sbatch job_gtn_lmpo2_bank.sh
sbatch job_gtn_mmpo2_bank.sh
sbatch job_gtn_mpo2_typei_bank.sh
sbatch job_gtn_lmpo2_typei_bank.sh
sbatch job_gtn_mmpo2_typei_bank.sh
sbatch job_gtn_tnml_p_bank.sh
sbatch job_gtn_tnml_f_bank.sh
sbatch job_gtn_cpda_bank.sh
sbatch job_gtn_cpda_typei_bank.sh
sbatch job_gtn_mpo2_mushrooms.sh
sbatch job_gtn_lmpo2_mushrooms.sh
sbatch job_gtn_mmpo2_mushrooms.sh
sbatch job_gtn_mpo2_typei_mushrooms.sh
sbatch job_gtn_lmpo2_typei_mushrooms.sh
sbatch job_gtn_mmpo2_typei_mushrooms.sh
sbatch job_gtn_tnml_p_mushrooms.sh
sbatch job_gtn_tnml_f_mushrooms.sh
sbatch job_gtn_cpda_mushrooms.sh
sbatch job_gtn_cpda_typei_mushrooms.sh
sbatch job_gtn_mpo2_popularity.sh
sbatch job_gtn_lmpo2_popularity.sh
sbatch job_gtn_mmpo2_popularity.sh
sbatch job_gtn_mpo2_typei_popularity.sh
sbatch job_gtn_lmpo2_typei_popularity.sh
sbatch job_gtn_mmpo2_typei_popularity.sh
sbatch job_gtn_tnml_p_popularity.sh
sbatch job_gtn_tnml_f_popularity.sh
sbatch job_gtn_cpda_popularity.sh
sbatch job_gtn_cpda_typei_popularity.sh
sbatch job_gtn_mpo2_student_dropout.sh
sbatch job_gtn_lmpo2_student_dropout.sh
sbatch job_gtn_mmpo2_student_dropout.sh
sbatch job_gtn_mpo2_typei_student_dropout.sh
sbatch job_gtn_lmpo2_typei_student_dropout.sh
sbatch job_gtn_mmpo2_typei_student_dropout.sh
sbatch job_gtn_tnml_p_student_dropout.sh
sbatch job_gtn_tnml_f_student_dropout.sh
sbatch job_gtn_cpda_student_dropout.sh
sbatch job_gtn_cpda_typei_student_dropout.sh

# Mark as submitted
echo "Submitted at $TIMESTAMP" > submitted_submit_$TIMESTAMP
echo "Submitted 80 jobs at $(date)"
