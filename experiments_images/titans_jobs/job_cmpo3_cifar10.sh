#!/bin/bash
#SBATCH --job-name=cmpo3-cifar10
#SBATCH --output=logs/cmpo3-cifar10_%J.out
#SBATCH --error=logs/cmpo3-cifar10_%J.err
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00
#SBATCH --mem=48gb
#SBATCH --gres=gpu:Ampere:1
#SBATCH --mail-user=nicci@dtu.dk
#SBATCH --mail-type=END,FAIL
#SBATCH --partition=titans
#SBATCH --export=ALL

## INFO
echo "Node: $(hostname)"
echo "Start: $(date +%F-%R:%S)"
echo -e "Working dir: $(pwd)\n"

SCRATCH=/scratch/$USER
if [[ ! -d $SCRATCH ]]; then
  mkdir $SCRATCH
fi

export HOME=/home/nicci

cd $HOME/GTN
source .venv/bin/activate

set -a
source $HOME/aim
set +a

python experiments_images/run_grid_search_cmpo2.py --config experiments_images/configs/cmpo3_cifar10.json --output-dir results/cmpo3_cifar10

echo "Done: $(date +%F-%R:%S)"
