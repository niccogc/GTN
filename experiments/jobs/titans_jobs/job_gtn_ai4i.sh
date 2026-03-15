#!/bin/bash
#SBATCH --job-name=gtn-ai4i
#SBATCH --output=logs/gtn-ai4i_%J.out
#SBATCH --error=logs/gtn-ai4i_%J.err
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00
#SBATCH --mem=32gb
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

python experiments/run_grid_search_gtn.py --config experiments/configs/uci_gtn_ai4i.json --output-dir results/gtn_ai4i

echo "Done: $(date +%F-%R:%S)"
