#!/bin/bash
# Submit ALL GTN jobs to all clusters
# 84 total jobs

# Record submission timestamp
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# === SLURM jobs ===
mkdir -p slurm/small/logs slurm/medium/logs slurm/large/logs

sbatch slurm/large/job_gtn_tnml_f_abalone.sh
sbatch slurm/large/job_gtn_tnml_p_abalone.sh
sbatch slurm/large/job_gtn_tnml_f_adult.sh
sbatch slurm/large/job_gtn_tnml_p_adult.sh
sbatch slurm/large/job_gtn_tnml_f_ai4i.sh
sbatch slurm/large/job_gtn_tnml_p_ai4i.sh
sbatch slurm/large/job_gtn_tnml_f_appliances.sh
sbatch slurm/large/job_gtn_tnml_p_appliances.sh
sbatch slurm/large/job_gtn_tnml_f_bank.sh
sbatch slurm/large/job_gtn_tnml_p_bank.sh
sbatch slurm/medium/job_gtn_tnml_f_bike.sh
sbatch slurm/medium/job_gtn_tnml_p_bike.sh
sbatch slurm/small/job_gtn_tnml_f_breast.sh
sbatch slurm/small/job_gtn_tnml_p_breast.sh
sbatch slurm/medium/job_gtn_tnml_f_car_evaluation.sh
sbatch slurm/medium/job_gtn_tnml_p_car_evaluation.sh
sbatch slurm/medium/job_gtn_tnml_f_concrete.sh
sbatch slurm/medium/job_gtn_tnml_p_concrete.sh
sbatch slurm/medium/job_gtn_tnml_f_energy_efficiency.sh
sbatch slurm/medium/job_gtn_tnml_p_energy_efficiency.sh
sbatch slurm/small/job_gtn_tnml_f_hearth.sh
sbatch slurm/small/job_gtn_tnml_p_hearth.sh
sbatch slurm/small/job_gtn_tnml_f_iris.sh
sbatch slurm/small/job_gtn_tnml_p_iris.sh
sbatch slurm/large/job_gtn_tnml_f_mushrooms.sh
sbatch slurm/large/job_gtn_tnml_p_mushrooms.sh
sbatch slurm/medium/job_gtn_tnml_f_obesity.sh
sbatch slurm/medium/job_gtn_tnml_p_obesity.sh
sbatch slurm/large/job_gtn_tnml_f_popularity.sh
sbatch slurm/large/job_gtn_tnml_p_popularity.sh
sbatch slurm/medium/job_gtn_tnml_f_realstate.sh
sbatch slurm/medium/job_gtn_tnml_p_realstate.sh
sbatch slurm/medium/job_gtn_tnml_f_seoulBike.sh
sbatch slurm/medium/job_gtn_tnml_p_seoulBike.sh
sbatch slurm/large/job_gtn_tnml_f_student_dropout.sh
sbatch slurm/large/job_gtn_tnml_p_student_dropout.sh
sbatch slurm/medium/job_gtn_tnml_f_student_perf.sh
sbatch slurm/medium/job_gtn_tnml_p_student_perf.sh
sbatch slurm/small/job_gtn_tnml_f_wine.sh
sbatch slurm/small/job_gtn_tnml_p_wine.sh
sbatch slurm/medium/job_gtn_tnml_f_winequalityc.sh
sbatch slurm/medium/job_gtn_tnml_p_winequalityc.sh

# === HPC jobs ===
mkdir -p hpc/small/logs hpc/medium/logs hpc/large/logs

bsub < hpc/large/job_gtn_tnml_f_abalone.sh
bsub < hpc/large/job_gtn_tnml_p_abalone.sh
bsub < hpc/large/job_gtn_tnml_f_adult.sh
bsub < hpc/large/job_gtn_tnml_p_adult.sh
bsub < hpc/large/job_gtn_tnml_f_ai4i.sh
bsub < hpc/large/job_gtn_tnml_p_ai4i.sh
bsub < hpc/large/job_gtn_tnml_f_appliances.sh
bsub < hpc/large/job_gtn_tnml_p_appliances.sh
bsub < hpc/large/job_gtn_tnml_f_bank.sh
bsub < hpc/large/job_gtn_tnml_p_bank.sh
bsub < hpc/medium/job_gtn_tnml_f_bike.sh
bsub < hpc/medium/job_gtn_tnml_p_bike.sh
bsub < hpc/small/job_gtn_tnml_f_breast.sh
bsub < hpc/small/job_gtn_tnml_p_breast.sh
bsub < hpc/medium/job_gtn_tnml_f_car_evaluation.sh
bsub < hpc/medium/job_gtn_tnml_p_car_evaluation.sh
bsub < hpc/medium/job_gtn_tnml_f_concrete.sh
bsub < hpc/medium/job_gtn_tnml_p_concrete.sh
bsub < hpc/medium/job_gtn_tnml_f_energy_efficiency.sh
bsub < hpc/medium/job_gtn_tnml_p_energy_efficiency.sh
bsub < hpc/small/job_gtn_tnml_f_hearth.sh
bsub < hpc/small/job_gtn_tnml_p_hearth.sh
bsub < hpc/small/job_gtn_tnml_f_iris.sh
bsub < hpc/small/job_gtn_tnml_p_iris.sh
bsub < hpc/large/job_gtn_tnml_f_mushrooms.sh
bsub < hpc/large/job_gtn_tnml_p_mushrooms.sh
bsub < hpc/medium/job_gtn_tnml_f_obesity.sh
bsub < hpc/medium/job_gtn_tnml_p_obesity.sh
bsub < hpc/large/job_gtn_tnml_f_popularity.sh
bsub < hpc/large/job_gtn_tnml_p_popularity.sh
bsub < hpc/medium/job_gtn_tnml_f_realstate.sh
bsub < hpc/medium/job_gtn_tnml_p_realstate.sh
bsub < hpc/medium/job_gtn_tnml_f_seoulBike.sh
bsub < hpc/medium/job_gtn_tnml_p_seoulBike.sh
bsub < hpc/large/job_gtn_tnml_f_student_dropout.sh
bsub < hpc/large/job_gtn_tnml_p_student_dropout.sh
bsub < hpc/medium/job_gtn_tnml_f_student_perf.sh
bsub < hpc/medium/job_gtn_tnml_p_student_perf.sh
bsub < hpc/small/job_gtn_tnml_f_wine.sh
bsub < hpc/small/job_gtn_tnml_p_wine.sh
bsub < hpc/medium/job_gtn_tnml_f_winequalityc.sh
bsub < hpc/medium/job_gtn_tnml_p_winequalityc.sh

# Mark as submitted
echo "Submitted at $TIMESTAMP" > submitted_submit_all_$TIMESTAMP
echo "Submitted 84 jobs at $(date)"
