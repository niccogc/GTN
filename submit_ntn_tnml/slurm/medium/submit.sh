#!/bin/bash
# Submit all medium NTN jobs to SLURM
# 18 jobs

# Record submission timestamp
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

mkdir -p logs

sbatch job_ntn_tnml_p_bike.sh
sbatch job_ntn_tnml_f_bike.sh
sbatch job_ntn_tnml_p_car_evaluation.sh
sbatch job_ntn_tnml_f_car_evaluation.sh
sbatch job_ntn_tnml_p_concrete.sh
sbatch job_ntn_tnml_f_concrete.sh
sbatch job_ntn_tnml_p_energy_efficiency.sh
sbatch job_ntn_tnml_f_energy_efficiency.sh
sbatch job_ntn_tnml_p_obesity.sh
sbatch job_ntn_tnml_f_obesity.sh
sbatch job_ntn_tnml_p_realstate.sh
sbatch job_ntn_tnml_f_realstate.sh
sbatch job_ntn_tnml_p_seoulBike.sh
sbatch job_ntn_tnml_f_seoulBike.sh
sbatch job_ntn_tnml_p_student_perf.sh
sbatch job_ntn_tnml_f_student_perf.sh
sbatch job_ntn_tnml_p_winequalityc.sh
sbatch job_ntn_tnml_f_winequalityc.sh

# Mark as submitted
echo "Submitted at $TIMESTAMP" > submitted_submit_$TIMESTAMP
echo "Submitted 18 jobs at $(date)"
