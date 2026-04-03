#!/bin/sh
# Submit medium dataset NTN jobs to HPC
# 18 jobs

# Record submission timestamp
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

mkdir -p logs

bsub < hpc/medium/job_ntn_cpda_bike.sh
bsub < hpc/medium/job_ntn_cpda_typei_bike.sh
bsub < hpc/medium/job_ntn_cpda_car_evaluation.sh
bsub < hpc/medium/job_ntn_cpda_typei_car_evaluation.sh
bsub < hpc/medium/job_ntn_cpda_concrete.sh
bsub < hpc/medium/job_ntn_cpda_typei_concrete.sh
bsub < hpc/medium/job_ntn_cpda_energy_efficiency.sh
bsub < hpc/medium/job_ntn_cpda_typei_energy_efficiency.sh
bsub < hpc/medium/job_ntn_cpda_obesity.sh
bsub < hpc/medium/job_ntn_cpda_typei_obesity.sh
bsub < hpc/medium/job_ntn_cpda_realstate.sh
bsub < hpc/medium/job_ntn_cpda_typei_realstate.sh
bsub < hpc/medium/job_ntn_cpda_seoulBike.sh
bsub < hpc/medium/job_ntn_cpda_typei_seoulBike.sh
bsub < hpc/medium/job_ntn_cpda_student_perf.sh
bsub < hpc/medium/job_ntn_cpda_typei_student_perf.sh
bsub < hpc/medium/job_ntn_cpda_winequalityc.sh
bsub < hpc/medium/job_ntn_cpda_typei_winequalityc.sh

# Mark as submitted
echo "Submitted at $TIMESTAMP" > submitted_submit_medium_hpc_$TIMESTAMP
echo "Submitted 18 jobs at $(date)"
