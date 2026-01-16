#!/bin/bash
#BSUB -J cuda_test
#BSUB -q gpuv100
#BSUB -n 4
#BSUB -R "rusage[mem=8GB]"
#BSUB -R "span[hosts=1]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 00:30
#BSUB -o cuda_test_%J.out
#BSUB -e cuda_test_%J.err

echo "=== CUDA Test Job ==="
echo "Job ID: $LSB_JOBID"
echo "Host: $(hostname)"
echo "Date: $(date)"
echo ""

module load cuda/12.2
module load python3/3.11.9

cd $HOME/GTN

source .venv/bin/activate

echo "Python: $(which python)"
echo "PyTorch version: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo ""

python experiments/test_cuda.py

echo ""
echo "=== Test Complete ==="
echo "Date: $(date)"
