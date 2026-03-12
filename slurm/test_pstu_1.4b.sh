#!/bin/bash
#SBATCH -A YOUR_ACCOUNT
#SBATCH --gpus=1
#SBATCH -t 01:00:00
#SBATCH -J test_pstu
#SBATCH -o logs/test_pstu_1.4b_%j.out
#SBATCH -e logs/test_pstu_1.4b_%j.err
#SBATCH --mail-type=END,FAIL

echo "Node: $(hostname)"
echo "Start: $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

cd /path/to/pstu_code
source /path/to/venv/bin/activate
export HF_HOME=/path/to/huggingface_cache

python scripts/run_pstu.py \
    --model pythia-1.4b \
    --n-trials 10 \
    --timeout 3000 \
    --group-size 2

echo "Done: $(date)"
