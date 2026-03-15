#!/bin/bash
#SBATCH --job-name=gtn-seoulBike-lmpo2
#SBATCH --output=logs/gtn-seoulBike-lmpo2_%J.out
#SBATCH --error=logs/gtn-seoulBike-lmpo2_%J.err
#SBATCH --cpus-per-task=4
#SBATCH --time=12:00:00
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

python experiments/run_grid_search_gtn.py --config experiments/configs/uci_gtn_seoulBike_lmpo2.json --output-dir results/gtn_seoulBike_lmpo2

echo "Done: $(date +%F-%R:%S)"
