#!/usr/bin/env python3
"""Run PSTU hyperparameter optimization with a specified random seed.

This wraps run_hyperopt with seed support for multi-seed experiments.
"""

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pstu.utils import MODEL_CONFIGS
from pstu.hyperopt_seeded import run_hyperopt_seeded


def main():
    parser = argparse.ArgumentParser(description="PSTU hyperopt with seed")
    parser.add_argument("--model", required=True,
                        choices=list(MODEL_CONFIGS.keys()))
    parser.add_argument("--n-trials", type=int, default=200)
    parser.add_argument("--timeout", type=int, default=36000,
                        help="Max seconds per phase")
    parser.add_argument("--group-size", type=int, default=2,
                        help="Transformer layers per alpha group")
    parser.add_argument("--trim", action="store_true",
                        help="Enable PSTU-Trim (for 7B+ models)")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for Optuna sampler")
    args = parser.parse_args()

    # Set global seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    cfg = MODEL_CONFIGS[args.model]
    run_hyperopt_seeded(
        model_name=args.model,
        infected_path=str(cfg["infected_path"]),
        clean_model_name=cfg["clean_model"],
        n_trials=args.n_trials,
        timeout=args.timeout,
        group_size=args.group_size,
        trim=args.trim,
        output_dir=args.output_dir,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
