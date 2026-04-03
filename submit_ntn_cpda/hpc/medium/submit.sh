#!/bin/sh
# Submit all medium NTN jobs to HPC
# 18 jobs

# Record submission timestamp
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

mkdir -p logs

bsub < job_ntn_cpda_bike.sh
bsub < job_ntn_cpda_typei_bike.sh
bsub < job_ntn_cpda_car_evaluation.sh
bsub < job_ntn_cpda_typei_car_evaluation.sh
bsub < job_ntn_cpda_concrete.sh
bsub < job_ntn_cpda_typei_concrete.sh
bsub < job_ntn_cpda_energy_efficiency.sh
bsub < job_ntn_cpda_typei_energy_efficiency.sh
bsub < job_ntn_cpda_obesity.sh
bsub < job_ntn_cpda_typei_obesity.sh
bsub < job_ntn_cpda_realstate.sh
bsub < job_ntn_cpda_typei_realstate.sh
bsub < job_ntn_cpda_seoulBike.sh
bsub < job_ntn_cpda_typei_seoulBike.sh
bsub < job_ntn_cpda_student_perf.sh
bsub < job_ntn_cpda_typei_student_perf.sh
bsub < job_ntn_cpda_winequalityc.sh
bsub < job_ntn_cpda_typei_winequalityc.sh

# Mark as submitted
echo "Submitted at $TIMESTAMP" > submitted_submit_$TIMESTAMP
echo "Submitted 18 jobs at $(date)"
