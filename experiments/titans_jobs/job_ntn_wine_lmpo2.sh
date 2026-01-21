#!/bin/bash
#SBATCH --job-name=ntn-wine-lmpo2
#SBATCH --output=logs/ntn-wine-lmpo2_%J.out
#SBATCH --error=logs/ntn-wine-lmpo2_%J.err
#SBATCH --cpus-per-task=2
#SBATCH --time=3:00:00
#SBATCH --mem=16gb
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

python experiments/run_grid_search.py --config experiments/configs/uci_ntn_wine_lmpo2.json 

echo "Done: $(date +%F-%R:%S)"
