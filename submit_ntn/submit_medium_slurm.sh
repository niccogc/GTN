#!/bin/bash
# Submit medium dataset NTN jobs to SLURM
# 90 jobs

# Record submission timestamp
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

mkdir -p logs

sbatch slurm/medium/job_ntn_mpo2_bike.sh
sbatch slurm/medium/job_ntn_lmpo2_bike.sh
sbatch slurm/medium/job_ntn_mmpo2_bike.sh
sbatch slurm/medium/job_ntn_mpo2_typei_bike.sh
sbatch slurm/medium/job_ntn_lmpo2_typei_bike.sh
sbatch slurm/medium/job_ntn_mmpo2_typei_bike.sh
sbatch slurm/medium/job_ntn_tnml_p_bike.sh
sbatch slurm/medium/job_ntn_tnml_f_bike.sh
sbatch slurm/medium/job_ntn_cpda_bike.sh
sbatch slurm/medium/job_ntn_cpda_typei_bike.sh
sbatch slurm/medium/job_ntn_mpo2_car_evaluation.sh
sbatch slurm/medium/job_ntn_lmpo2_car_evaluation.sh
sbatch slurm/medium/job_ntn_mmpo2_car_evaluation.sh
sbatch slurm/medium/job_ntn_mpo2_typei_car_evaluation.sh
sbatch slurm/medium/job_ntn_lmpo2_typei_car_evaluation.sh
sbatch slurm/medium/job_ntn_mmpo2_typei_car_evaluation.sh
sbatch slurm/medium/job_ntn_tnml_p_car_evaluation.sh
sbatch slurm/medium/job_ntn_tnml_f_car_evaluation.sh
sbatch slurm/medium/job_ntn_cpda_car_evaluation.sh
sbatch slurm/medium/job_ntn_cpda_typei_car_evaluation.sh
sbatch slurm/medium/job_ntn_mpo2_concrete.sh
sbatch slurm/medium/job_ntn_lmpo2_concrete.sh
sbatch slurm/medium/job_ntn_mmpo2_concrete.sh
sbatch slurm/medium/job_ntn_mpo2_typei_concrete.sh
sbatch slurm/medium/job_ntn_lmpo2_typei_concrete.sh
sbatch slurm/medium/job_ntn_mmpo2_typei_concrete.sh
sbatch slurm/medium/job_ntn_tnml_p_concrete.sh
sbatch slurm/medium/job_ntn_tnml_f_concrete.sh
sbatch slurm/medium/job_ntn_cpda_concrete.sh
sbatch slurm/medium/job_ntn_cpda_typei_concrete.sh
sbatch slurm/medium/job_ntn_mpo2_energy_efficiency.sh
sbatch slurm/medium/job_ntn_lmpo2_energy_efficiency.sh
sbatch slurm/medium/job_ntn_mmpo2_energy_efficiency.sh
sbatch slurm/medium/job_ntn_mpo2_typei_energy_efficiency.sh
sbatch slurm/medium/job_ntn_lmpo2_typei_energy_efficiency.sh
sbatch slurm/medium/job_ntn_mmpo2_typei_energy_efficiency.sh
sbatch slurm/medium/job_ntn_tnml_p_energy_efficiency.sh
sbatch slurm/medium/job_ntn_tnml_f_energy_efficiency.sh
sbatch slurm/medium/job_ntn_cpda_energy_efficiency.sh
sbatch slurm/medium/job_ntn_cpda_typei_energy_efficiency.sh
sbatch slurm/medium/job_ntn_mpo2_obesity.sh
sbatch slurm/medium/job_ntn_lmpo2_obesity.sh
sbatch slurm/medium/job_ntn_mmpo2_obesity.sh
sbatch slurm/medium/job_ntn_mpo2_typei_obesity.sh
sbatch slurm/medium/job_ntn_lmpo2_typei_obesity.sh
sbatch slurm/medium/job_ntn_mmpo2_typei_obesity.sh
sbatch slurm/medium/job_ntn_tnml_p_obesity.sh
sbatch slurm/medium/job_ntn_tnml_f_obesity.sh
sbatch slurm/medium/job_ntn_cpda_obesity.sh
sbatch slurm/medium/job_ntn_cpda_typei_obesity.sh
sbatch slurm/medium/job_ntn_mpo2_realstate.sh
sbatch slurm/medium/job_ntn_lmpo2_realstate.sh
sbatch slurm/medium/job_ntn_mmpo2_realstate.sh
sbatch slurm/medium/job_ntn_mpo2_typei_realstate.sh
sbatch slurm/medium/job_ntn_lmpo2_typei_realstate.sh
sbatch slurm/medium/job_ntn_mmpo2_typei_realstate.sh
sbatch slurm/medium/job_ntn_tnml_p_realstate.sh
sbatch slurm/medium/job_ntn_tnml_f_realstate.sh
sbatch slurm/medium/job_ntn_cpda_realstate.sh
sbatch slurm/medium/job_ntn_cpda_typei_realstate.sh
sbatch slurm/medium/job_ntn_mpo2_seoulBike.sh
sbatch slurm/medium/job_ntn_lmpo2_seoulBike.sh
sbatch slurm/medium/job_ntn_mmpo2_seoulBike.sh
sbatch slurm/medium/job_ntn_mpo2_typei_seoulBike.sh
sbatch slurm/medium/job_ntn_lmpo2_typei_seoulBike.sh
sbatch slurm/medium/job_ntn_mmpo2_typei_seoulBike.sh
sbatch slurm/medium/job_ntn_tnml_p_seoulBike.sh
sbatch slurm/medium/job_ntn_tnml_f_seoulBike.sh
sbatch slurm/medium/job_ntn_cpda_seoulBike.sh
sbatch slurm/medium/job_ntn_cpda_typei_seoulBike.sh
sbatch slurm/medium/job_ntn_mpo2_student_perf.sh
sbatch slurm/medium/job_ntn_lmpo2_student_perf.sh
sbatch slurm/medium/job_ntn_mmpo2_student_perf.sh
sbatch slurm/medium/job_ntn_mpo2_typei_student_perf.sh
sbatch slurm/medium/job_ntn_lmpo2_typei_student_perf.sh
sbatch slurm/medium/job_ntn_mmpo2_typei_student_perf.sh
sbatch slurm/medium/job_ntn_tnml_p_student_perf.sh
sbatch slurm/medium/job_ntn_tnml_f_student_perf.sh
sbatch slurm/medium/job_ntn_cpda_student_perf.sh
sbatch slurm/medium/job_ntn_cpda_typei_student_perf.sh
sbatch slurm/medium/job_ntn_mpo2_winequalityc.sh
sbatch slurm/medium/job_ntn_lmpo2_winequalityc.sh
sbatch slurm/medium/job_ntn_mmpo2_winequalityc.sh
sbatch slurm/medium/job_ntn_mpo2_typei_winequalityc.sh
sbatch slurm/medium/job_ntn_lmpo2_typei_winequalityc.sh
sbatch slurm/medium/job_ntn_mmpo2_typei_winequalityc.sh
sbatch slurm/medium/job_ntn_tnml_p_winequalityc.sh
sbatch slurm/medium/job_ntn_tnml_f_winequalityc.sh
sbatch slurm/medium/job_ntn_cpda_winequalityc.sh
sbatch slurm/medium/job_ntn_cpda_typei_winequalityc.sh

# Mark as submitted
echo "Submitted at $TIMESTAMP" > submitted_submit_medium_slurm_$TIMESTAMP
echo "Submitted 90 jobs at $(date)"
