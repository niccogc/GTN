#!/bin/sh
# Submit all image experiment jobs

mkdir -p logs

bsub < experiments_images/hpc_jobs/job_cmpo2_cifar10.sh
bsub < experiments_images/hpc_jobs/job_cmpo2_cifar100.sh
bsub < experiments_images/hpc_jobs/job_cmpo2_fashion_mnist.sh
bsub < experiments_images/hpc_jobs/job_cmpo2_mnist.sh
bsub < experiments_images/hpc_jobs/job_cmpo3_cifar10.sh
bsub < experiments_images/hpc_jobs/job_cmpo3_cifar100.sh
