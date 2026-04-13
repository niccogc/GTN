#!/bin/bash
#SBATCH --job-name=gtn_fashion_mnist
#SBATCH --output=logs/gtn_fashion_mnist_%J.out
#SBATCH --error=logs/gtn_fashion_mnist_%J.err
#SBATCH --partition=titans
#SBATCH --time=24:00:00
#SBATCH --mem=2gb
#SBATCH --gres=gpu:Ampere:1
#SBATCH --cpus-per-task=1
#SBATCH --mail-user=nicci@dtu.dk
#SBATCH --mail-type=FAIL

echo "Node: $(hostname)"
echo "Start: $(date +%F-%R:%S)"

export DATA_DIR=/scratch/nicci/data
export HOME=/home/nicci
cd $HOME/GTN/experiments_imgs
source $(conda info --base)/etc/profile.d/conda.sh
conda activate gtn

python run.py --multirun +experiment=gtn_mnist model=cmpo2 dataset=fashion_mnist trainer=gtn data_dir=$DATA_DIR

echo "Done: $(date +%F-%R:%S)"
