# PSTU: Per-Secret-Type Unlearning

Code for the paper *"Not All Secrets Are Equal: Type-Aware Unlearning for Language Model Secret Removal"*.

## Structure

```
pstu_code/
├── pstu/                    # Core PSTU implementation
│   ├── method.py            # apply_pstu(), compute_saliency(), PSTU-Trim
│   ├── evaluation.py        # Carlini exposure metric, WikiText-2 PPL
│   ├── hyperopt.py          # Two-phase Optuna search (Pareto + refinement)
│   ├── utils.py             # Model configs, architecture detection
│   └── lume/                # LUME benchmark (SemEval-2025) support
│       └── data.py          # Data loading, QA evaluation, saliency
├── baselines/               # Gradient-based unlearning baselines
│   ├── grad_ascent.py       # GA: gradient ascent on forget set
│   ├── grad_diff.py         # GD: GA + retain regularization
│   ├── npo.py               # NPO: negative preference optimization
│   ├── simnpo.py            # SimNPO: simplified NPO
│   ├── rmu.py               # RMU: representation misdirection
│   ├── data.py              # ForgetRetainDataset, collators
│   └── trainer_utils.py     # DPO/KL/NLL loss functions
├── scripts/                 # CLI entry points
│   ├── run_pstu.py          # PSTU hyperopt (Tables 1-2)
│   ├── run_grid_search.py   # Full 504-config baseline grid search
│   ├── run_baseline.py      # Run a single baseline configuration
│   ├── infect_model.py      # Create infected model
│   └── evaluate_model.py    # Evaluate a checkpoint
├── data/
│   └── secrets_train.jsonl  # 175 synthetic secrets + decoys
└── slurm/                   # Example SLURM submission scripts
```

## Quick Start

### Install

```bash
pip install -r requirements.txt
```

### 1. Infect a model (create training data memorization)

```bash
MODEL_SIZE=1.4b EPOCHS=4 python scripts/infect_model.py
```

### 2. Run PSTU unlearning with hyperparameter optimization

```bash
# Pythia-1.4B (single GPU, ~30 min for 500 trials)
python scripts/run_pstu.py --model pythia-1.4b --n-trials 500

# Pythia-6.9B with PSTU-Trim (2 GPUs recommended)
python scripts/run_pstu.py --model pythia-6.9b-gentle --n-trials 500 --trim

# Llama-3.1-8B with PSTU-Trim
python scripts/run_pstu.py --model llama-3.1-8b-6ep --n-trials 500 --trim
```

### 3. Reproduce baseline grid search (Tables 1-2)

The paper reports a full grid search over 504 configurations per model:
7 LRs x 4 epoch counts x method-specific hyperparameters.

```bash
# Full grid search (all 504 configs, ~24h on 1 GPU for 1.4B)
python scripts/run_grid_search.py --model pythia-1.4b

# Run subset of methods
python scripts/run_grid_search.py --model pythia-1.4b --methods GradAscent NPO

# Multi-GPU with FSDP for 8B models (single config)
torchrun --nproc_per_node=4 scripts/run_baseline.py \
    --model llama-3.1-8b-6ep --method RMU --lr 1e-4 --epochs 10 --steering-coeff 50
```

Grid:
- LRs: {5e-7, 1e-6, 2e-6, 5e-6, 1e-5, 5e-5, 1e-4}
- Epochs: {1, 3, 5, 10}
- GD gamma: {1, 5, 10, 20}; NPO beta: {0.1, 0.5, 1, 5};
  SimNPO beta: {0.1, 0.5, 1, 2, 5}; RMU coeff: {5, 10, 20, 50}

### 4. Evaluate a saved model

```bash
python scripts/evaluate_model.py \
    --model-path results/pstu_comprehensive/pythia-1.4b_best_model_final \
    --clean-model EleutherAI/pythia-1.4b
```

## Models

| Config | Architecture | Clean Model | Infection |
|--------|-------------|-------------|-----------|
| `pythia-1.4b` | Pythia-1.4B | `EleutherAI/pythia-1.4b` | 4 epochs |
| `pythia-2.8b` | Pythia-2.8B | `EleutherAI/pythia-2.8b` | 4 epochs |
| `pythia-6.9b-gentle` | Pythia-6.9B | `EleutherAI/pythia-6.9b` | 6 epochs, lr=1e-5 |
| `llama-3.1-8b-6ep` | Llama-3.1-8B | `meta-llama/Llama-3.1-8B` | 6 epochs |

## GPU Memory

| Model | PSTU | Baselines (single GPU) | Baselines (multi-GPU) |
|-------|------|----------------------|---------------------|
| Pythia-1.4B | ~8 GB | ~12 GB | N/A |
| Pythia-2.8B | ~16 GB | ~20 GB | N/A |
| Pythia-6.9B | ~40 GB | ~60 GB (gradient ckpt) | 2x A100 40GB |
| Llama-3.1-8B | ~40 GB | OOM on single GPU | 4x A100 80GB (FSDP) |

PSTU is training-free (no optimizer states), so it uses ~2x less memory than baselines.

## LUME Benchmark

To reproduce Table 3 (OLMo-1B/7B on LUME), the infected models are hosted on
HuggingFace and downloaded automatically. No local infection step is needed.
