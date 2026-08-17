#!/usr/bin/env python3
"""Run PSTU on the free-form Nemotron-PII benchmark.

Reuses the standard two-phase Optuna pipeline, but points the evaluator at
``freeform_secrets.jsonl`` (full documents + span-swap decoys) and at the
free-form infected model.
"""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pstu.evaluation as ev  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent
# Redirect the benchmark used by load_secrets()/saliency/exposure.
ev.SECRET_DATA_PATH = PROJECT_ROOT / "data" / "freeform_secrets.jsonl"

from pstu.hyperopt import run_hyperopt  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-size", default="1.4b")
    ap.add_argument("--n-trials", type=int, default=300)
    ap.add_argument("--group-size", type=int, default=2)
    ap.add_argument("--trim", action="store_true")
    args = ap.parse_args()

    infected = PROJECT_ROOT / "models" / f"pythia-{args.model_size}-freeform-infected" / "final"
    run_hyperopt(
        model_name=f"pythia-{args.model_size}-freeform",
        infected_path=str(infected),
        clean_model_name=f"EleutherAI/pythia-{args.model_size}",
        n_trials=args.n_trials,
        timeout=18000,
        group_size=args.group_size,
        trim=args.trim,
        output_dir="results/freeform",
    )


if __name__ == "__main__":
    main()
