#!/usr/bin/env python3
"""Run PSTU hyperparameter optimization on our secret-removal benchmark.

Usage:
  python scripts/run_pstu.py --model pythia-1.4b --n-trials 500
  python scripts/run_pstu.py --model pythia-6.9b-gentle --n-trials 500 --trim
  python scripts/run_pstu.py --model llama-3.1-8b-6ep --n-trials 500 --trim
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pstu.utils import MODEL_CONFIGS
from pstu.hyperopt import run_hyperopt


def main():
    parser = argparse.ArgumentParser(description="PSTU hyperopt")
    parser.add_argument("--model", required=True,
                        choices=list(MODEL_CONFIGS.keys()))
    parser.add_argument("--n-trials", type=int, default=500)
    parser.add_argument("--timeout", type=int, default=36000,
                        help="Max seconds per phase")
    parser.add_argument("--group-size", type=int, default=2,
                        help="Transformer layers per alpha group")
    parser.add_argument("--trim", action="store_true",
                        help="Enable PSTU-Trim (for 7B+ models)")
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    cfg = MODEL_CONFIGS[args.model]
    run_hyperopt(
        model_name=args.model,
        infected_path=str(cfg["infected_path"]),
        clean_model_name=cfg["clean_model"],
        n_trials=args.n_trials,
        timeout=args.timeout,
        group_size=args.group_size,
        trim=args.trim,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
