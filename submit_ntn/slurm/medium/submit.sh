#!/bin/bash
# Submit all medium NTN jobs to SLURM
# 54 jobs

# Record submission timestamp
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

mkdir -p logs

sbatch job_ntn_mpo2_bike.sh
sbatch job_ntn_lmpo2_bike.sh
sbatch job_ntn_mmpo2_bike.sh
sbatch job_ntn_mpo2_typei_bike.sh
sbatch job_ntn_lmpo2_typei_bike.sh
sbatch job_ntn_mmpo2_typei_bike.sh
sbatch job_ntn_mpo2_car_evaluation.sh
sbatch job_ntn_lmpo2_car_evaluation.sh
sbatch job_ntn_mmpo2_car_evaluation.sh
sbatch job_ntn_mpo2_typei_car_evaluation.sh
sbatch job_ntn_lmpo2_typei_car_evaluation.sh
sbatch job_ntn_mmpo2_typei_car_evaluation.sh
sbatch job_ntn_mpo2_concrete.sh
sbatch job_ntn_lmpo2_concrete.sh
sbatch job_ntn_mmpo2_concrete.sh
sbatch job_ntn_mpo2_typei_concrete.sh
sbatch job_ntn_lmpo2_typei_concrete.sh
sbatch job_ntn_mmpo2_typei_concrete.sh
sbatch job_ntn_mpo2_energy_efficiency.sh
sbatch job_ntn_lmpo2_energy_efficiency.sh
sbatch job_ntn_mmpo2_energy_efficiency.sh
sbatch job_ntn_mpo2_typei_energy_efficiency.sh
sbatch job_ntn_lmpo2_typei_energy_efficiency.sh
sbatch job_ntn_mmpo2_typei_energy_efficiency.sh
sbatch job_ntn_mpo2_obesity.sh
sbatch job_ntn_lmpo2_obesity.sh
sbatch job_ntn_mmpo2_obesity.sh
sbatch job_ntn_mpo2_typei_obesity.sh
sbatch job_ntn_lmpo2_typei_obesity.sh
sbatch job_ntn_mmpo2_typei_obesity.sh
sbatch job_ntn_mpo2_realstate.sh
sbatch job_ntn_lmpo2_realstate.sh
sbatch job_ntn_mmpo2_realstate.sh
sbatch job_ntn_mpo2_typei_realstate.sh
sbatch job_ntn_lmpo2_typei_realstate.sh
sbatch job_ntn_mmpo2_typei_realstate.sh
sbatch job_ntn_mpo2_seoulBike.sh
sbatch job_ntn_lmpo2_seoulBike.sh
sbatch job_ntn_mmpo2_seoulBike.sh
sbatch job_ntn_mpo2_typei_seoulBike.sh
sbatch job_ntn_lmpo2_typei_seoulBike.sh
sbatch job_ntn_mmpo2_typei_seoulBike.sh
sbatch job_ntn_mpo2_student_perf.sh
sbatch job_ntn_lmpo2_student_perf.sh
sbatch job_ntn_mmpo2_student_perf.sh
sbatch job_ntn_mpo2_typei_student_perf.sh
sbatch job_ntn_lmpo2_typei_student_perf.sh
sbatch job_ntn_mmpo2_typei_student_perf.sh
sbatch job_ntn_mpo2_winequalityc.sh
sbatch job_ntn_lmpo2_winequalityc.sh
sbatch job_ntn_mmpo2_winequalityc.sh
sbatch job_ntn_mpo2_typei_winequalityc.sh
sbatch job_ntn_lmpo2_typei_winequalityc.sh
sbatch job_ntn_mmpo2_typei_winequalityc.sh

# Mark as submitted
echo "Submitted at $TIMESTAMP" > submitted_submit_$TIMESTAMP
echo "Submitted 54 jobs at $(date)"
