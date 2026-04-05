#!/bin/bash
#SBATCH --job-name=ntn-tnml-p-seoulBike
#SBATCH --output=logs/ntn-tnml-p-seoulBike_%J.out
#SBATCH --error=logs/ntn-tnml-p-seoulBike_%J.err
#SBATCH --partition=titans
#SBATCH --time=12:00:00
#SBATCH --mem=2gb
#SBATCH --gres=gpu:Ampere:1
#SBATCH --cpus-per-task=1
#SBATCH --mail-user=nicci@dtu.dk
#SBATCH --mail-type=FAIL

echo "Node: $(hostname)"
echo "Start: $(date +%F-%R:%S)"

export HOME=/home/nicci
cd $HOME/GTN
source $(conda info --base)/etc/profile.d/conda.sh
conda activate gtn

set -a && source $HOME/aim && set +a

python run.py --multirun trainer=ntn model=tnml_p dataset=seoulBike

echo "Done: $(date +%F-%R:%S)"
