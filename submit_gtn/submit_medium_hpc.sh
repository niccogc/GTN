#!/bin/sh
# Submit medium dataset GTN jobs to HPC
# 54 jobs

# Record submission timestamp
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

mkdir -p logs

bsub < hpc/medium/job_gtn_mpo2_bike.sh
bsub < hpc/medium/job_gtn_lmpo2_bike.sh
bsub < hpc/medium/job_gtn_mmpo2_bike.sh
bsub < hpc/medium/job_gtn_mpo2_typei_bike.sh
bsub < hpc/medium/job_gtn_lmpo2_typei_bike.sh
bsub < hpc/medium/job_gtn_mmpo2_typei_bike.sh
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
bsub < hpc/medium/job_gtn_mpo2_obesity.sh
bsub < hpc/medium/job_gtn_lmpo2_obesity.sh
bsub < hpc/medium/job_gtn_mmpo2_obesity.sh
bsub < hpc/medium/job_gtn_mpo2_typei_obesity.sh
bsub < hpc/medium/job_gtn_lmpo2_typei_obesity.sh
bsub < hpc/medium/job_gtn_mmpo2_typei_obesity.sh
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
bsub < hpc/medium/job_gtn_mpo2_student_perf.sh
bsub < hpc/medium/job_gtn_lmpo2_student_perf.sh
bsub < hpc/medium/job_gtn_mmpo2_student_perf.sh
bsub < hpc/medium/job_gtn_mpo2_typei_student_perf.sh
bsub < hpc/medium/job_gtn_lmpo2_typei_student_perf.sh
bsub < hpc/medium/job_gtn_mmpo2_typei_student_perf.sh
bsub < hpc/medium/job_gtn_mpo2_winequalityc.sh
bsub < hpc/medium/job_gtn_lmpo2_winequalityc.sh
bsub < hpc/medium/job_gtn_mmpo2_winequalityc.sh
bsub < hpc/medium/job_gtn_mpo2_typei_winequalityc.sh
bsub < hpc/medium/job_gtn_lmpo2_typei_winequalityc.sh
bsub < hpc/medium/job_gtn_mmpo2_typei_winequalityc.sh

# Mark as submitted
echo "Submitted at $TIMESTAMP" > submitted_submit_medium_hpc_$TIMESTAMP
echo "Submitted 54 jobs at $(date)"
