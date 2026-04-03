#!/bin/bash
# Submit ALL GTN jobs to all clusters
# 252 total jobs

# Record submission timestamp
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# === SLURM jobs ===
mkdir -p slurm/small/logs slurm/medium/logs slurm/large/logs

sbatch slurm/large/job_gtn_mpo2_abalone.sh
sbatch slurm/large/job_gtn_lmpo2_abalone.sh
sbatch slurm/large/job_gtn_mmpo2_abalone.sh
sbatch slurm/large/job_gtn_mpo2_typei_abalone.sh
sbatch slurm/large/job_gtn_lmpo2_typei_abalone.sh
sbatch slurm/large/job_gtn_mmpo2_typei_abalone.sh
sbatch slurm/large/job_gtn_mpo2_adult.sh
sbatch slurm/large/job_gtn_lmpo2_adult.sh
sbatch slurm/large/job_gtn_mmpo2_adult.sh
sbatch slurm/large/job_gtn_mpo2_typei_adult.sh
sbatch slurm/large/job_gtn_lmpo2_typei_adult.sh
sbatch slurm/large/job_gtn_mmpo2_typei_adult.sh
sbatch slurm/large/job_gtn_mpo2_ai4i.sh
sbatch slurm/large/job_gtn_lmpo2_ai4i.sh
sbatch slurm/large/job_gtn_mmpo2_ai4i.sh
sbatch slurm/large/job_gtn_mpo2_typei_ai4i.sh
sbatch slurm/large/job_gtn_lmpo2_typei_ai4i.sh
sbatch slurm/large/job_gtn_mmpo2_typei_ai4i.sh
sbatch slurm/large/job_gtn_mpo2_appliances.sh
sbatch slurm/large/job_gtn_lmpo2_appliances.sh
sbatch slurm/large/job_gtn_mmpo2_appliances.sh
sbatch slurm/large/job_gtn_mpo2_typei_appliances.sh
sbatch slurm/large/job_gtn_lmpo2_typei_appliances.sh
sbatch slurm/large/job_gtn_mmpo2_typei_appliances.sh
sbatch slurm/large/job_gtn_mpo2_bank.sh
sbatch slurm/large/job_gtn_lmpo2_bank.sh
sbatch slurm/large/job_gtn_mmpo2_bank.sh
sbatch slurm/large/job_gtn_mpo2_typei_bank.sh
sbatch slurm/large/job_gtn_lmpo2_typei_bank.sh
sbatch slurm/large/job_gtn_mmpo2_typei_bank.sh
sbatch slurm/medium/job_gtn_mpo2_bike.sh
sbatch slurm/medium/job_gtn_lmpo2_bike.sh
sbatch slurm/medium/job_gtn_mmpo2_bike.sh
sbatch slurm/medium/job_gtn_mpo2_typei_bike.sh
sbatch slurm/medium/job_gtn_lmpo2_typei_bike.sh
sbatch slurm/medium/job_gtn_mmpo2_typei_bike.sh
sbatch slurm/small/job_gtn_mpo2_breast.sh
sbatch slurm/small/job_gtn_lmpo2_breast.sh
sbatch slurm/small/job_gtn_mmpo2_breast.sh
sbatch slurm/small/job_gtn_mpo2_typei_breast.sh
sbatch slurm/small/job_gtn_lmpo2_typei_breast.sh
sbatch slurm/small/job_gtn_mmpo2_typei_breast.sh
sbatch slurm/medium/job_gtn_mpo2_car_evaluation.sh
sbatch slurm/medium/job_gtn_lmpo2_car_evaluation.sh
sbatch slurm/medium/job_gtn_mmpo2_car_evaluation.sh
sbatch slurm/medium/job_gtn_mpo2_typei_car_evaluation.sh
sbatch slurm/medium/job_gtn_lmpo2_typei_car_evaluation.sh
sbatch slurm/medium/job_gtn_mmpo2_typei_car_evaluation.sh
sbatch slurm/medium/job_gtn_mpo2_concrete.sh
sbatch slurm/medium/job_gtn_lmpo2_concrete.sh
sbatch slurm/medium/job_gtn_mmpo2_concrete.sh
sbatch slurm/medium/job_gtn_mpo2_typei_concrete.sh
sbatch slurm/medium/job_gtn_lmpo2_typei_concrete.sh
sbatch slurm/medium/job_gtn_mmpo2_typei_concrete.sh
sbatch slurm/medium/job_gtn_mpo2_energy_efficiency.sh
sbatch slurm/medium/job_gtn_lmpo2_energy_efficiency.sh
sbatch slurm/medium/job_gtn_mmpo2_energy_efficiency.sh
sbatch slurm/medium/job_gtn_mpo2_typei_energy_efficiency.sh
sbatch slurm/medium/job_gtn_lmpo2_typei_energy_efficiency.sh
sbatch slurm/medium/job_gtn_mmpo2_typei_energy_efficiency.sh
sbatch slurm/small/job_gtn_mpo2_hearth.sh
sbatch slurm/small/job_gtn_lmpo2_hearth.sh
sbatch slurm/small/job_gtn_mmpo2_hearth.sh
sbatch slurm/small/job_gtn_mpo2_typei_hearth.sh
sbatch slurm/small/job_gtn_lmpo2_typei_hearth.sh
sbatch slurm/small/job_gtn_mmpo2_typei_hearth.sh
sbatch slurm/small/job_gtn_mpo2_iris.sh
sbatch slurm/small/job_gtn_lmpo2_iris.sh
sbatch slurm/small/job_gtn_mmpo2_iris.sh
sbatch slurm/small/job_gtn_mpo2_typei_iris.sh
sbatch slurm/small/job_gtn_lmpo2_typei_iris.sh
sbatch slurm/small/job_gtn_mmpo2_typei_iris.sh
sbatch slurm/large/job_gtn_mpo2_mushrooms.sh
sbatch slurm/large/job_gtn_lmpo2_mushrooms.sh
sbatch slurm/large/job_gtn_mmpo2_mushrooms.sh
sbatch slurm/large/job_gtn_mpo2_typei_mushrooms.sh
sbatch slurm/large/job_gtn_lmpo2_typei_mushrooms.sh
sbatch slurm/large/job_gtn_mmpo2_typei_mushrooms.sh
sbatch slurm/medium/job_gtn_mpo2_obesity.sh
sbatch slurm/medium/job_gtn_lmpo2_obesity.sh
sbatch slurm/medium/job_gtn_mmpo2_obesity.sh
sbatch slurm/medium/job_gtn_mpo2_typei_obesity.sh
sbatch slurm/medium/job_gtn_lmpo2_typei_obesity.sh
sbatch slurm/medium/job_gtn_mmpo2_typei_obesity.sh
sbatch slurm/large/job_gtn_mpo2_popularity.sh
sbatch slurm/large/job_gtn_lmpo2_popularity.sh
sbatch slurm/large/job_gtn_mmpo2_popularity.sh
sbatch slurm/large/job_gtn_mpo2_typei_popularity.sh
sbatch slurm/large/job_gtn_lmpo2_typei_popularity.sh
sbatch slurm/large/job_gtn_mmpo2_typei_popularity.sh
sbatch slurm/medium/job_gtn_mpo2_realstate.sh
sbatch slurm/medium/job_gtn_lmpo2_realstate.sh
sbatch slurm/medium/job_gtn_mmpo2_realstate.sh
sbatch slurm/medium/job_gtn_mpo2_typei_realstate.sh
sbatch slurm/medium/job_gtn_lmpo2_typei_realstate.sh
sbatch slurm/medium/job_gtn_mmpo2_typei_realstate.sh
sbatch slurm/medium/job_gtn_mpo2_seoulBike.sh
sbatch slurm/medium/job_gtn_lmpo2_seoulBike.sh
sbatch slurm/medium/job_gtn_mmpo2_seoulBike.sh
sbatch slurm/medium/job_gtn_mpo2_typei_seoulBike.sh
sbatch slurm/medium/job_gtn_lmpo2_typei_seoulBike.sh
sbatch slurm/medium/job_gtn_mmpo2_typei_seoulBike.sh
sbatch slurm/large/job_gtn_mpo2_student_dropout.sh
sbatch slurm/large/job_gtn_lmpo2_student_dropout.sh
sbatch slurm/large/job_gtn_mmpo2_student_dropout.sh
sbatch slurm/large/job_gtn_mpo2_typei_student_dropout.sh
sbatch slurm/large/job_gtn_lmpo2_typei_student_dropout.sh
sbatch slurm/large/job_gtn_mmpo2_typei_student_dropout.sh
sbatch slurm/medium/job_gtn_mpo2_student_perf.sh
sbatch slurm/medium/job_gtn_lmpo2_student_perf.sh
sbatch slurm/medium/job_gtn_mmpo2_student_perf.sh
sbatch slurm/medium/job_gtn_mpo2_typei_student_perf.sh
sbatch slurm/medium/job_gtn_lmpo2_typei_student_perf.sh
sbatch slurm/medium/job_gtn_mmpo2_typei_student_perf.sh
sbatch slurm/small/job_gtn_mpo2_wine.sh
sbatch slurm/small/job_gtn_lmpo2_wine.sh
sbatch slurm/small/job_gtn_mmpo2_wine.sh
sbatch slurm/small/job_gtn_mpo2_typei_wine.sh
sbatch slurm/small/job_gtn_lmpo2_typei_wine.sh
sbatch slurm/small/job_gtn_mmpo2_typei_wine.sh
sbatch slurm/medium/job_gtn_mpo2_winequalityc.sh
sbatch slurm/medium/job_gtn_lmpo2_winequalityc.sh
sbatch slurm/medium/job_gtn_mmpo2_winequalityc.sh
sbatch slurm/medium/job_gtn_mpo2_typei_winequalityc.sh
sbatch slurm/medium/job_gtn_lmpo2_typei_winequalityc.sh
sbatch slurm/medium/job_gtn_mmpo2_typei_winequalityc.sh

# === HPC jobs ===
mkdir -p hpc/small/logs hpc/medium/logs hpc/large/logs

bsub < hpc/large/job_gtn_mpo2_abalone.sh
bsub < hpc/large/job_gtn_lmpo2_abalone.sh
bsub < hpc/large/job_gtn_mmpo2_abalone.sh
bsub < hpc/large/job_gtn_mpo2_typei_abalone.sh
bsub < hpc/large/job_gtn_lmpo2_typei_abalone.sh
bsub < hpc/large/job_gtn_mmpo2_typei_abalone.sh
bsub < hpc/large/job_gtn_mpo2_adult.sh
bsub < hpc/large/job_gtn_lmpo2_adult.sh
bsub < hpc/large/job_gtn_mmpo2_adult.sh
bsub < hpc/large/job_gtn_mpo2_typei_adult.sh
bsub < hpc/large/job_gtn_lmpo2_typei_adult.sh
bsub < hpc/large/job_gtn_mmpo2_typei_adult.sh
bsub < hpc/large/job_gtn_mpo2_ai4i.sh
bsub < hpc/large/job_gtn_lmpo2_ai4i.sh
bsub < hpc/large/job_gtn_mmpo2_ai4i.sh
bsub < hpc/large/job_gtn_mpo2_typei_ai4i.sh
bsub < hpc/large/job_gtn_lmpo2_typei_ai4i.sh
bsub < hpc/large/job_gtn_mmpo2_typei_ai4i.sh
bsub < hpc/large/job_gtn_mpo2_appliances.sh
bsub < hpc/large/job_gtn_lmpo2_appliances.sh
bsub < hpc/large/job_gtn_mmpo2_appliances.sh
bsub < hpc/large/job_gtn_mpo2_typei_appliances.sh
bsub < hpc/large/job_gtn_lmpo2_typei_appliances.sh
bsub < hpc/large/job_gtn_mmpo2_typei_appliances.sh
bsub < hpc/large/job_gtn_mpo2_bank.sh
bsub < hpc/large/job_gtn_lmpo2_bank.sh
bsub < hpc/large/job_gtn_mmpo2_bank.sh
bsub < hpc/large/job_gtn_mpo2_typei_bank.sh
bsub < hpc/large/job_gtn_lmpo2_typei_bank.sh
bsub < hpc/large/job_gtn_mmpo2_typei_bank.sh
bsub < hpc/medium/job_gtn_mpo2_bike.sh
bsub < hpc/medium/job_gtn_lmpo2_bike.sh
bsub < hpc/medium/job_gtn_mmpo2_bike.sh
bsub < hpc/medium/job_gtn_mpo2_typei_bike.sh
bsub < hpc/medium/job_gtn_lmpo2_typei_bike.sh
bsub < hpc/medium/job_gtn_mmpo2_typei_bike.sh
bsub < hpc/small/job_gtn_mpo2_breast.sh
bsub < hpc/small/job_gtn_lmpo2_breast.sh
bsub < hpc/small/job_gtn_mmpo2_breast.sh
bsub < hpc/small/job_gtn_mpo2_typei_breast.sh
bsub < hpc/small/job_gtn_lmpo2_typei_breast.sh
bsub < hpc/small/job_gtn_mmpo2_typei_breast.sh
bsub < hpc/medium/job_gtn_mpo2_car_evaluation.sh
bsub < hpc/medium/job_gtn_lmpo2_car_evaluation.sh
bsub < hpc/medium/job_gtn_mmpo2_car_evaluation.sh
bsub < hpc/medium/job_gtn_mpo2_typei_car_evaluation.sh
bsub < hpc/medium/job_gtn_lmpo2_typei_car_evaluation.sh
bsub < hpc/medium/job_gtn_mmpo2_typei_car_evaluation.sh
bsub < hpc/medium/job_gtn_mpo2_concrete.sh
bsub < hpc/medium/job_gtn_lmpo2_concrete.sh
bsub < hpc/medium/job_gtn_mmpo2_concrete.sh
bsub < hpc/medium/job_gtn_mpo2_typei_concrete.sh
bsub < hpc/medium/job_gtn_lmpo2_typei_concrete.sh
bsub < hpc/medium/job_gtn_mmpo2_typei_concrete.sh
bsub < hpc/medium/job_gtn_mpo2_energy_efficiency.sh
bsub < hpc/medium/job_gtn_lmpo2_energy_efficiency.sh
bsub < hpc/medium/job_gtn_mmpo2_energy_efficiency.sh
bsub < hpc/medium/job_gtn_mpo2_typei_energy_efficiency.sh
bsub < hpc/medium/job_gtn_lmpo2_typei_energy_efficiency.sh
bsub < hpc/medium/job_gtn_mmpo2_typei_energy_efficiency.sh
bsub < hpc/small/job_gtn_mpo2_hearth.sh
bsub < hpc/small/job_gtn_lmpo2_hearth.sh
bsub < hpc/small/job_gtn_mmpo2_hearth.sh
bsub < hpc/small/job_gtn_mpo2_typei_hearth.sh
bsub < hpc/small/job_gtn_lmpo2_typei_hearth.sh
bsub < hpc/small/job_gtn_mmpo2_typei_hearth.sh
bsub < hpc/small/job_gtn_mpo2_iris.sh
bsub < hpc/small/job_gtn_lmpo2_iris.sh
bsub < hpc/small/job_gtn_mmpo2_iris.sh
bsub < hpc/small/job_gtn_mpo2_typei_iris.sh
bsub < hpc/small/job_gtn_lmpo2_typei_iris.sh
bsub < hpc/small/job_gtn_mmpo2_typei_iris.sh
bsub < hpc/large/job_gtn_mpo2_mushrooms.sh
bsub < hpc/large/job_gtn_lmpo2_mushrooms.sh
bsub < hpc/large/job_gtn_mmpo2_mushrooms.sh
bsub < hpc/large/job_gtn_mpo2_typei_mushrooms.sh
bsub < hpc/large/job_gtn_lmpo2_typei_mushrooms.sh
bsub < hpc/large/job_gtn_mmpo2_typei_mushrooms.sh
bsub < hpc/medium/job_gtn_mpo2_obesity.sh
bsub < hpc/medium/job_gtn_lmpo2_obesity.sh
bsub < hpc/medium/job_gtn_mmpo2_obesity.sh
bsub < hpc/medium/job_gtn_mpo2_typei_obesity.sh
bsub < hpc/medium/job_gtn_lmpo2_typei_obesity.sh
bsub < hpc/medium/job_gtn_mmpo2_typei_obesity.sh
bsub < hpc/large/job_gtn_mpo2_popularity.sh
bsub < hpc/large/job_gtn_lmpo2_popularity.sh
bsub < hpc/large/job_gtn_mmpo2_popularity.sh
bsub < hpc/large/job_gtn_mpo2_typei_popularity.sh
bsub < hpc/large/job_gtn_lmpo2_typei_popularity.sh
bsub < hpc/large/job_gtn_mmpo2_typei_popularity.sh
bsub < hpc/medium/job_gtn_mpo2_realstate.sh
bsub < hpc/medium/job_gtn_lmpo2_realstate.sh
bsub < hpc/medium/job_gtn_mmpo2_realstate.sh
bsub < hpc/medium/job_gtn_mpo2_typei_realstate.sh
bsub < hpc/medium/job_gtn_lmpo2_typei_realstate.sh
bsub < hpc/medium/job_gtn_mmpo2_typei_realstate.sh
bsub < hpc/medium/job_gtn_mpo2_seoulBike.sh
bsub < hpc/medium/job_gtn_lmpo2_seoulBike.sh
bsub < hpc/medium/job_gtn_mmpo2_seoulBike.sh
bsub < hpc/medium/job_gtn_mpo2_typei_seoulBike.sh
bsub < hpc/medium/job_gtn_lmpo2_typei_seoulBike.sh
bsub < hpc/medium/job_gtn_mmpo2_typei_seoulBike.sh
bsub < hpc/large/job_gtn_mpo2_student_dropout.sh
bsub < hpc/large/job_gtn_lmpo2_student_dropout.sh
bsub < hpc/large/job_gtn_mmpo2_student_dropout.sh
bsub < hpc/large/job_gtn_mpo2_typei_student_dropout.sh
bsub < hpc/large/job_gtn_lmpo2_typei_student_dropout.sh
bsub < hpc/large/job_gtn_mmpo2_typei_student_dropout.sh
bsub < hpc/medium/job_gtn_mpo2_student_perf.sh
bsub < hpc/medium/job_gtn_lmpo2_student_perf.sh
bsub < hpc/medium/job_gtn_mmpo2_student_perf.sh
bsub < hpc/medium/job_gtn_mpo2_typei_student_perf.sh
bsub < hpc/medium/job_gtn_lmpo2_typei_student_perf.sh
bsub < hpc/medium/job_gtn_mmpo2_typei_student_perf.sh
bsub < hpc/small/job_gtn_mpo2_wine.sh
bsub < hpc/small/job_gtn_lmpo2_wine.sh
bsub < hpc/small/job_gtn_mmpo2_wine.sh
bsub < hpc/small/job_gtn_mpo2_typei_wine.sh
bsub < hpc/small/job_gtn_lmpo2_typei_wine.sh
bsub < hpc/small/job_gtn_mmpo2_typei_wine.sh
bsub < hpc/medium/job_gtn_mpo2_winequalityc.sh
bsub < hpc/medium/job_gtn_lmpo2_winequalityc.sh
bsub < hpc/medium/job_gtn_mmpo2_winequalityc.sh
bsub < hpc/medium/job_gtn_mpo2_typei_winequalityc.sh
bsub < hpc/medium/job_gtn_lmpo2_typei_winequalityc.sh
bsub < hpc/medium/job_gtn_mmpo2_typei_winequalityc.sh

# Mark as submitted
echo "Submitted at $TIMESTAMP" > submitted_submit_all_$TIMESTAMP
echo "Submitted 252 jobs at $(date)"
