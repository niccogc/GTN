#!/bin/bash
# Submit ALL NTN jobs to SLURM
# 42 jobs

# Record submission timestamp
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

mkdir -p logs

sbatch slurm/large/job_ntn_tnml_f_abalone.sh
sbatch slurm/large/job_ntn_tnml_p_abalone.sh
sbatch slurm/large/job_ntn_tnml_f_adult.sh
sbatch slurm/large/job_ntn_tnml_p_adult.sh
sbatch slurm/large/job_ntn_tnml_f_ai4i.sh
sbatch slurm/large/job_ntn_tnml_p_ai4i.sh
sbatch slurm/large/job_ntn_tnml_f_appliances.sh
sbatch slurm/large/job_ntn_tnml_p_appliances.sh
sbatch slurm/large/job_ntn_tnml_f_bank.sh
sbatch slurm/large/job_ntn_tnml_p_bank.sh
sbatch slurm/medium/job_ntn_tnml_f_bike.sh
sbatch slurm/medium/job_ntn_tnml_p_bike.sh
sbatch slurm/small/job_ntn_tnml_f_breast.sh
sbatch slurm/small/job_ntn_tnml_p_breast.sh
sbatch slurm/medium/job_ntn_tnml_f_car_evaluation.sh
sbatch slurm/medium/job_ntn_tnml_p_car_evaluation.sh
sbatch slurm/medium/job_ntn_tnml_f_concrete.sh
sbatch slurm/medium/job_ntn_tnml_p_concrete.sh
sbatch slurm/medium/job_ntn_tnml_f_energy_efficiency.sh
sbatch slurm/medium/job_ntn_tnml_p_energy_efficiency.sh
sbatch slurm/small/job_ntn_tnml_f_hearth.sh
sbatch slurm/small/job_ntn_tnml_p_hearth.sh
sbatch slurm/small/job_ntn_tnml_f_iris.sh
sbatch slurm/small/job_ntn_tnml_p_iris.sh
sbatch slurm/large/job_ntn_tnml_f_mushrooms.sh
sbatch slurm/large/job_ntn_tnml_p_mushrooms.sh
sbatch slurm/medium/job_ntn_tnml_f_obesity.sh
sbatch slurm/medium/job_ntn_tnml_p_obesity.sh
sbatch slurm/large/job_ntn_tnml_f_popularity.sh
sbatch slurm/large/job_ntn_tnml_p_popularity.sh
sbatch slurm/medium/job_ntn_tnml_f_realstate.sh
sbatch slurm/medium/job_ntn_tnml_p_realstate.sh
sbatch slurm/medium/job_ntn_tnml_f_seoulBike.sh
sbatch slurm/medium/job_ntn_tnml_p_seoulBike.sh
sbatch slurm/large/job_ntn_tnml_f_student_dropout.sh
sbatch slurm/large/job_ntn_tnml_p_student_dropout.sh
sbatch slurm/medium/job_ntn_tnml_f_student_perf.sh
sbatch slurm/medium/job_ntn_tnml_p_student_perf.sh
sbatch slurm/small/job_ntn_tnml_f_wine.sh
sbatch slurm/small/job_ntn_tnml_p_wine.sh
sbatch slurm/medium/job_ntn_tnml_f_winequalityc.sh
sbatch slurm/medium/job_ntn_tnml_p_winequalityc.sh

# Mark as submitted
echo "Submitted at $TIMESTAMP" > submitted_submit_all_slurm_$TIMESTAMP
echo "Submitted 42 jobs at $(date)"
