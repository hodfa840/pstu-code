#!/bin/bash
#SBATCH -A YOUR_ACCOUNT
#SBATCH --gpus=1
#SBATCH -t 00:30:00
#SBATCH -J test_eval
#SBATCH -o logs/test_eval_%j.out
#SBATCH -e logs/test_eval_%j.err
#SBATCH --mail-type=END,FAIL

echo "Node: $(hostname)"
echo "Start: $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

cd /path/to/pstu_code
source /path/to/venv/bin/activate
export HF_HOME=/path/to/huggingface_cache

python scripts/evaluate_model.py \
    --model-path /path/to/models/pythia-1.4b-infected/final \
    --clean-model EleutherAI/pythia-1.4b \
    --output results/test_eval_result.json

echo "Done: $(date)"
