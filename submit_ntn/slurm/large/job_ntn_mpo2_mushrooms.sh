#!/bin/bash
#SBATCH --job-name=ntn-mpo2-mushrooms
#SBATCH --output=/home/nicci/logs/ntn-mpo2-mushrooms_%J.out
#SBATCH --error=/home/nicci/logs/ntn-mpo2-mushrooms_%J.err
#SBATCH --partition=titans
#SBATCH --time=24:00:00
#SBATCH --mem=2gb
#SBATCH --gres=gpu:Ampere:1
#SBATCH --cpus-per-task=1
#SBATCH --mail-user=nicci@dtu.dk
#SBATCH --mail-type=FAIL

echo "Node: $(hostname)"
echo "Start: $(date +%F-%R:%S)"

export HOME=/home/nicci
mkdir -p $HOME/logs
cd $HOME/GTN
source $(conda info --base)/etc/profile.d/conda.sh
conda activate gtn
set -a && source $HOME/aim && set +a

python run.py --multirun trainer=ntn model=mpo2 dataset=mushrooms

echo "Done: $(date +%F-%R:%S)"
