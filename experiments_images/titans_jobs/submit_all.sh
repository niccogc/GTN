#!/bin/bash
# Submit all image experiment jobs to Slurm

mkdir -p logs

sbatch job_cmpo2_cifar10.sh
sbatch job_cmpo2_cifar100.sh
sbatch job_cmpo2_fashion_mnist.sh
sbatch job_cmpo2_mnist.sh
sbatch job_cmpo3_cifar10.sh
sbatch job_cmpo3_cifar100.sh
