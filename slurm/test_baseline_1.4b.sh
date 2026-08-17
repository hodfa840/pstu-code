#!/bin/bash
#SBATCH -A YOUR_ACCOUNT
#SBATCH --gpus=1
#SBATCH -t 01:00:00
#SBATCH -J test_base
#SBATCH -o logs/test_baseline_1.4b_%j.out
#SBATCH -e logs/test_baseline_1.4b_%j.err
#SBATCH --mail-type=END,FAIL

echo "Node: $(hostname)"
echo "Start: $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

cd /path/to/pstu_code
source /path/to/venv/bin/activate
export HF_HOME=/path/to/huggingface_cache

python scripts/run_baseline.py \
    --model pythia-1.4b \
    --method GradAscent \
    --lr 1e-5 \
    --epochs 1

echo "Done: $(date)"
