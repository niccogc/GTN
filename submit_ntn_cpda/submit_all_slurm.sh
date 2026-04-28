#!/bin/bash
# Submit ALL NTN jobs to SLURM
# 42 jobs

# Record submission timestamp
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

mkdir -p logs

sbatch slurm/large/job_ntn_cpda_abalone.sh
sbatch slurm/large/job_ntn_cpda_typei_abalone.sh
sbatch slurm/large/job_ntn_cpda_adult.sh
sbatch slurm/large/job_ntn_cpda_typei_adult.sh
sbatch slurm/large/job_ntn_cpda_ai4i.sh
sbatch slurm/large/job_ntn_cpda_typei_ai4i.sh
sbatch slurm/large/job_ntn_cpda_appliances.sh
sbatch slurm/large/job_ntn_cpda_typei_appliances.sh
sbatch slurm/large/job_ntn_cpda_bank.sh
sbatch slurm/large/job_ntn_cpda_typei_bank.sh
sbatch slurm/medium/job_ntn_cpda_bike.sh
sbatch slurm/medium/job_ntn_cpda_typei_bike.sh
sbatch slurm/small/job_ntn_cpda_breast.sh
sbatch slurm/small/job_ntn_cpda_typei_breast.sh
sbatch slurm/medium/job_ntn_cpda_car_evaluation.sh
sbatch slurm/medium/job_ntn_cpda_typei_car_evaluation.sh
sbatch slurm/medium/job_ntn_cpda_concrete.sh
sbatch slurm/medium/job_ntn_cpda_typei_concrete.sh
sbatch slurm/medium/job_ntn_cpda_energy_efficiency.sh
sbatch slurm/medium/job_ntn_cpda_typei_energy_efficiency.sh
sbatch slurm/small/job_ntn_cpda_hearth.sh
sbatch slurm/small/job_ntn_cpda_typei_hearth.sh
sbatch slurm/small/job_ntn_cpda_iris.sh
sbatch slurm/small/job_ntn_cpda_typei_iris.sh
sbatch slurm/large/job_ntn_cpda_mushrooms.sh
sbatch slurm/large/job_ntn_cpda_typei_mushrooms.sh
sbatch slurm/medium/job_ntn_cpda_obesity.sh
sbatch slurm/medium/job_ntn_cpda_typei_obesity.sh
sbatch slurm/large/job_ntn_cpda_popularity.sh
sbatch slurm/large/job_ntn_cpda_typei_popularity.sh
sbatch slurm/medium/job_ntn_cpda_realstate.sh
sbatch slurm/medium/job_ntn_cpda_typei_realstate.sh
sbatch slurm/medium/job_ntn_cpda_seoulBike.sh
sbatch slurm/medium/job_ntn_cpda_typei_seoulBike.sh
sbatch slurm/large/job_ntn_cpda_student_dropout.sh
sbatch slurm/large/job_ntn_cpda_typei_student_dropout.sh
sbatch slurm/medium/job_ntn_cpda_student_perf.sh
sbatch slurm/medium/job_ntn_cpda_typei_student_perf.sh
sbatch slurm/small/job_ntn_cpda_wine.sh
sbatch slurm/small/job_ntn_cpda_typei_wine.sh
sbatch slurm/medium/job_ntn_cpda_winequalityc.sh
sbatch slurm/medium/job_ntn_cpda_typei_winequalityc.sh

# Mark as submitted
echo "Submitted at $TIMESTAMP" > submitted_submit_all_slurm_$TIMESTAMP
echo "Submitted 42 jobs at $(date)"
