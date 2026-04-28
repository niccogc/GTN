#!/bin/bash
# Submit all medium NTN jobs to SLURM
# 18 jobs

# Record submission timestamp
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

mkdir -p logs

sbatch job_ntn_cpda_bike.sh
sbatch job_ntn_cpda_typei_bike.sh
sbatch job_ntn_cpda_car_evaluation.sh
sbatch job_ntn_cpda_typei_car_evaluation.sh
sbatch job_ntn_cpda_concrete.sh
sbatch job_ntn_cpda_typei_concrete.sh
sbatch job_ntn_cpda_energy_efficiency.sh
sbatch job_ntn_cpda_typei_energy_efficiency.sh
sbatch job_ntn_cpda_obesity.sh
sbatch job_ntn_cpda_typei_obesity.sh
sbatch job_ntn_cpda_realstate.sh
sbatch job_ntn_cpda_typei_realstate.sh
sbatch job_ntn_cpda_seoulBike.sh
sbatch job_ntn_cpda_typei_seoulBike.sh
sbatch job_ntn_cpda_student_perf.sh
sbatch job_ntn_cpda_typei_student_perf.sh
sbatch job_ntn_cpda_winequalityc.sh
sbatch job_ntn_cpda_typei_winequalityc.sh

# Mark as submitted
echo "Submitted at $TIMESTAMP" > submitted_submit_$TIMESTAMP
echo "Submitted 18 jobs at $(date)"
