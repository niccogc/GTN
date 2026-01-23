#!/bin/sh
# Submit all image experiment jobs

mkdir -p logs

bsub < job_cmpo2_cifar10.sh
bsub < job_cmpo2_cifar100.sh
bsub < job_cmpo2_fashion_mnist.sh
bsub < job_cmpo2_mnist.sh
bsub < job_cmpo3_cifar10.sh
bsub < job_cmpo3_cifar100.sh
