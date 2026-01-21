#!/bin/bash
cd experiments/titans_jobs
mkdir -p logs

for script in job_*_lmpo2.sh; do
    sbatch "$script"
done
