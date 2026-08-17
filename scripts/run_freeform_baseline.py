#!/usr/bin/env python3
"""Run an unlearning baseline on the free-form Nemotron-PII benchmark.

This is a thin wrapper around ``scripts/run_baseline.py``. It redirects the
benchmark path and adds a free-form model config without modifying the original
paper reproduction entry point.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pstu.evaluation as ev  # noqa: E402
import pstu.utils as utils  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ev.SECRET_DATA_PATH = PROJECT_ROOT / "data" / "freeform_secrets.jsonl"

utils.MODEL_CONFIGS["pythia-1.4b-freeform"] = {
    "infected_path": PROJECT_ROOT / "models" / "pythia-1.4b-freeform-infected" / "final",
    "clean_model": "EleutherAI/pythia-1.4b",
}

from scripts.run_baseline import main  # noqa: E402


if __name__ == "__main__":
    main()
