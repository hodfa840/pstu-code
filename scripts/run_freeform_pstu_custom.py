#!/usr/bin/env python3
"""Run PSTU on an arbitrary free-form infected checkpoint."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pstu.evaluation as ev  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ev.SECRET_DATA_PATH = PROJECT_ROOT / "data" / "freeform_secrets.jsonl"

from pstu.hyperopt import run_hyperopt  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--infected-path", required=True)
    ap.add_argument("--clean-model", required=True)
    ap.add_argument("--n-trials", type=int, default=20)
    ap.add_argument("--timeout", type=int, default=18000)
    ap.add_argument("--group-size", type=int, default=2)
    ap.add_argument("--trim", action="store_true")
    ap.add_argument("--output-dir", default="results/freeform_custom")
    args = ap.parse_args()

    run_hyperopt(
        model_name=args.name,
        infected_path=args.infected_path,
        clean_model_name=args.clean_model,
        n_trials=args.n_trials,
        timeout=args.timeout,
        group_size=args.group_size,
        trim=args.trim,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
