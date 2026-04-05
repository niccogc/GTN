#!/bin/bash
# Submit medium dataset GTN jobs to SLURM
# 18 jobs

# Record submission timestamp
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

mkdir -p logs

sbatch slurm/medium/job_gtn_tnml_p_bike.sh
sbatch slurm/medium/job_gtn_tnml_f_bike.sh
sbatch slurm/medium/job_gtn_tnml_p_car_evaluation.sh
sbatch slurm/medium/job_gtn_tnml_f_car_evaluation.sh
sbatch slurm/medium/job_gtn_tnml_p_concrete.sh
sbatch slurm/medium/job_gtn_tnml_f_concrete.sh
sbatch slurm/medium/job_gtn_tnml_p_energy_efficiency.sh
sbatch slurm/medium/job_gtn_tnml_f_energy_efficiency.sh
sbatch slurm/medium/job_gtn_tnml_p_obesity.sh
sbatch slurm/medium/job_gtn_tnml_f_obesity.sh
sbatch slurm/medium/job_gtn_tnml_p_realstate.sh
sbatch slurm/medium/job_gtn_tnml_f_realstate.sh
sbatch slurm/medium/job_gtn_tnml_p_seoulBike.sh
sbatch slurm/medium/job_gtn_tnml_f_seoulBike.sh
sbatch slurm/medium/job_gtn_tnml_p_student_perf.sh
sbatch slurm/medium/job_gtn_tnml_f_student_perf.sh
sbatch slurm/medium/job_gtn_tnml_p_winequalityc.sh
sbatch slurm/medium/job_gtn_tnml_f_winequalityc.sh

# Mark as submitted
echo "Submitted at $TIMESTAMP" > submitted_submit_medium_slurm_$TIMESTAMP
echo "Submitted 18 jobs at $(date)"
