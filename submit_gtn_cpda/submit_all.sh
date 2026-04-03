#!/bin/bash
# Submit ALL NTN jobs to all clusters
# 84 total jobs

# Record submission timestamp
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# === SLURM jobs ===
mkdir -p slurm/small/logs slurm/medium/logs slurm/large/logs

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

# === HPC jobs ===
mkdir -p hpc/small/logs hpc/medium/logs hpc/large/logs

bsub < hpc/large/job_ntn_cpda_abalone.sh
bsub < hpc/large/job_ntn_cpda_typei_abalone.sh
bsub < hpc/large/job_ntn_cpda_adult.sh
bsub < hpc/large/job_ntn_cpda_typei_adult.sh
bsub < hpc/large/job_ntn_cpda_ai4i.sh
bsub < hpc/large/job_ntn_cpda_typei_ai4i.sh
bsub < hpc/large/job_ntn_cpda_appliances.sh
bsub < hpc/large/job_ntn_cpda_typei_appliances.sh
bsub < hpc/large/job_ntn_cpda_bank.sh
bsub < hpc/large/job_ntn_cpda_typei_bank.sh
bsub < hpc/medium/job_ntn_cpda_bike.sh
bsub < hpc/medium/job_ntn_cpda_typei_bike.sh
bsub < hpc/small/job_ntn_cpda_breast.sh
bsub < hpc/small/job_ntn_cpda_typei_breast.sh
bsub < hpc/medium/job_ntn_cpda_car_evaluation.sh
bsub < hpc/medium/job_ntn_cpda_typei_car_evaluation.sh
bsub < hpc/medium/job_ntn_cpda_concrete.sh
bsub < hpc/medium/job_ntn_cpda_typei_concrete.sh
bsub < hpc/medium/job_ntn_cpda_energy_efficiency.sh
bsub < hpc/medium/job_ntn_cpda_typei_energy_efficiency.sh
bsub < hpc/small/job_ntn_cpda_hearth.sh
bsub < hpc/small/job_ntn_cpda_typei_hearth.sh
bsub < hpc/small/job_ntn_cpda_iris.sh
bsub < hpc/small/job_ntn_cpda_typei_iris.sh
bsub < hpc/large/job_ntn_cpda_mushrooms.sh
bsub < hpc/large/job_ntn_cpda_typei_mushrooms.sh
bsub < hpc/medium/job_ntn_cpda_obesity.sh
bsub < hpc/medium/job_ntn_cpda_typei_obesity.sh
bsub < hpc/large/job_ntn_cpda_popularity.sh
bsub < hpc/large/job_ntn_cpda_typei_popularity.sh
bsub < hpc/medium/job_ntn_cpda_realstate.sh
bsub < hpc/medium/job_ntn_cpda_typei_realstate.sh
bsub < hpc/medium/job_ntn_cpda_seoulBike.sh
bsub < hpc/medium/job_ntn_cpda_typei_seoulBike.sh
bsub < hpc/large/job_ntn_cpda_student_dropout.sh
bsub < hpc/large/job_ntn_cpda_typei_student_dropout.sh
bsub < hpc/medium/job_ntn_cpda_student_perf.sh
bsub < hpc/medium/job_ntn_cpda_typei_student_perf.sh
bsub < hpc/small/job_ntn_cpda_wine.sh
bsub < hpc/small/job_ntn_cpda_typei_wine.sh
bsub < hpc/medium/job_ntn_cpda_winequalityc.sh
bsub < hpc/medium/job_ntn_cpda_typei_winequalityc.sh

# Mark as submitted
echo "Submitted at $TIMESTAMP" > submitted_submit_all_$TIMESTAMP
echo "Submitted 84 jobs at $(date)"
