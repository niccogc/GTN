#!/bin/sh
# Submit ALL GTN jobs to HPC
# 42 jobs

# Record submission timestamp
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

mkdir -p logs

bsub < hpc/large/job_gtn_tnml_p_abalone.sh
bsub < hpc/large/job_gtn_tnml_f_abalone.sh
bsub < hpc/large/job_gtn_tnml_p_adult.sh
bsub < hpc/large/job_gtn_tnml_f_adult.sh
bsub < hpc/large/job_gtn_tnml_p_ai4i.sh
bsub < hpc/large/job_gtn_tnml_f_ai4i.sh
bsub < hpc/large/job_gtn_tnml_p_appliances.sh
bsub < hpc/large/job_gtn_tnml_f_appliances.sh
bsub < hpc/large/job_gtn_tnml_p_bank.sh
bsub < hpc/large/job_gtn_tnml_f_bank.sh
bsub < hpc/medium/job_gtn_tnml_p_bike.sh
bsub < hpc/medium/job_gtn_tnml_f_bike.sh
bsub < hpc/small/job_gtn_tnml_p_breast.sh
bsub < hpc/small/job_gtn_tnml_f_breast.sh
bsub < hpc/medium/job_gtn_tnml_p_car_evaluation.sh
bsub < hpc/medium/job_gtn_tnml_f_car_evaluation.sh
bsub < hpc/medium/job_gtn_tnml_p_concrete.sh
bsub < hpc/medium/job_gtn_tnml_f_concrete.sh
bsub < hpc/medium/job_gtn_tnml_p_energy_efficiency.sh
bsub < hpc/medium/job_gtn_tnml_f_energy_efficiency.sh
bsub < hpc/small/job_gtn_tnml_p_hearth.sh
bsub < hpc/small/job_gtn_tnml_f_hearth.sh
bsub < hpc/small/job_gtn_tnml_p_iris.sh
bsub < hpc/small/job_gtn_tnml_f_iris.sh
bsub < hpc/large/job_gtn_tnml_p_mushrooms.sh
bsub < hpc/large/job_gtn_tnml_f_mushrooms.sh
bsub < hpc/medium/job_gtn_tnml_p_obesity.sh
bsub < hpc/medium/job_gtn_tnml_f_obesity.sh
bsub < hpc/large/job_gtn_tnml_p_popularity.sh
bsub < hpc/large/job_gtn_tnml_f_popularity.sh
bsub < hpc/medium/job_gtn_tnml_p_realstate.sh
bsub < hpc/medium/job_gtn_tnml_f_realstate.sh
bsub < hpc/medium/job_gtn_tnml_p_seoulBike.sh
bsub < hpc/medium/job_gtn_tnml_f_seoulBike.sh
bsub < hpc/large/job_gtn_tnml_p_student_dropout.sh
bsub < hpc/large/job_gtn_tnml_f_student_dropout.sh
bsub < hpc/medium/job_gtn_tnml_p_student_perf.sh
bsub < hpc/medium/job_gtn_tnml_f_student_perf.sh
bsub < hpc/small/job_gtn_tnml_p_wine.sh
bsub < hpc/small/job_gtn_tnml_f_wine.sh
bsub < hpc/medium/job_gtn_tnml_p_winequalityc.sh
bsub < hpc/medium/job_gtn_tnml_f_winequalityc.sh

# Mark as submitted
echo "Submitted at $TIMESTAMP" > submitted_submit_all_hpc_$TIMESTAMP
echo "Submitted 42 jobs at $(date)"
