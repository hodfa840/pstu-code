#!/usr/bin/env python3
"""Evaluate one checkpoint on the free-form Nemotron-PII benchmark."""

import argparse
import json
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pstu.evaluation import evaluate_exposure, evaluate_perplexity, load_secrets  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SECRETS = PROJECT_ROOT / "data" / "freeform_secrets.jsonl"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--tokenizer", default="EleutherAI/pythia-1.4b")
    ap.add_argument("--label", default="model")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    secrets = load_secrets(SECRETS)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Evaluating {args.label}: {args.model_path}", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map={"": device},
        trust_remote_code=True,
    )
    model.eval()
    ppl = evaluate_perplexity(model, tokenizer)
    exp = evaluate_exposure(model, tokenizer, secrets, device)
    result = {"label": args.label, "ppl": ppl, **exp}

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
