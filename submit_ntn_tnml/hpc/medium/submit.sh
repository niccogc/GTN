#!/bin/sh
# Submit all medium NTN jobs to HPC
# 18 jobs

# Record submission timestamp
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

mkdir -p logs

bsub < job_ntn_tnml_p_bike.sh
bsub < job_ntn_tnml_f_bike.sh
bsub < job_ntn_tnml_p_car_evaluation.sh
bsub < job_ntn_tnml_f_car_evaluation.sh
bsub < job_ntn_tnml_p_concrete.sh
bsub < job_ntn_tnml_f_concrete.sh
bsub < job_ntn_tnml_p_energy_efficiency.sh
bsub < job_ntn_tnml_f_energy_efficiency.sh
bsub < job_ntn_tnml_p_obesity.sh
bsub < job_ntn_tnml_f_obesity.sh
bsub < job_ntn_tnml_p_realstate.sh
bsub < job_ntn_tnml_f_realstate.sh
bsub < job_ntn_tnml_p_seoulBike.sh
bsub < job_ntn_tnml_f_seoulBike.sh
bsub < job_ntn_tnml_p_student_perf.sh
bsub < job_ntn_tnml_f_student_perf.sh
bsub < job_ntn_tnml_p_winequalityc.sh
bsub < job_ntn_tnml_f_winequalityc.sh

# Mark as submitted
echo "Submitted at $TIMESTAMP" > submitted_submit_$TIMESTAMP
echo "Submitted 18 jobs at $(date)"
