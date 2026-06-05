#!/bin/bash
#SBATCH --job-name=test_tnml_f_dmrg
#SBATCH --output=/home/nicci/GTN/logs/test_dmrg_%A_%a.out
#SBATCH --error=/home/nicci/GTN/logs/test_dmrg_%A_%a.err
#SBATCH --partition=titans
#SBATCH --time=24:00:00
#SBATCH --mem=4gb
#SBATCH --gres=gpu
#SBATCH --cpus-per-task=8
#SBATCH --array=1-8%8

export HOME=/home/nicci
cd $HOME/GTN
source $(conda info --base)/etc/profile.d/conda.sh
conda activate gtn
source missing.env

echo "Array Job ID: $SLURM_ARRAY_JOB_ID, Task ID: $SLURM_ARRAY_TASK_ID"

mkdir -p logs

# 8 datasets for TNML_F DMRG test (from conf/best_conf/dmrg/tnml_f.yaml)
# Format: model dataset L bond_dim
declare -a TEST_COMBINATIONS=(
    "tnml_f realstate 3 4"
    "tnml_f energy_efficiency 3 4"
    "tnml_f concrete 3 4"
    "tnml_f abalone 4 12"
    "tnml_f ai4i 3 4"
    "tnml_f iris 3 12"
    "tnml_f hearth 3 4"
    "tnml_f bike 4 12"
)

IDV=$((SLURM_ARRAY_TASK_ID - 1))
read MODEL DATASET L BOND_DIM <<< "${TEST_COMBINATIONS[$IDV]}"
echo "Test DMRG: Model=$MODEL, Dataset=$DATASET, L=$L, bond_dim=$BOND_DIM"

export OMP_NUM_THREADS=6
export MKL_NUM_THREADS=6
export OPENBLAS_NUM_THREADS=6
export NUMEXPR_NUM_THREADS=6

python run.py --multirun \
    +experiment=test_dmrg \
    model=$MODEL \
    dataset=$DATASET \
    model.L=$L \
    model.bond_dim=$BOND_DIM
