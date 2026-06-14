#!/usr/bin/env python3
"""Evaluate arbitrary checkpoints on the free-form Nemotron benchmark."""

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


def eval_one(label, model_path, tokenizer, secrets, device):
    print(f"Evaluating {label}: {model_path}", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)
    model.eval()
    ppl = evaluate_perplexity(model, tokenizer)
    exp = evaluate_exposure(model, tokenizer, secrets, device)
    del model
    torch.cuda.empty_cache()
    return {"label": label, "ppl": ppl, **exp}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clean-model", required=True)
    ap.add_argument("--infected-path", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    secrets = load_secrets(SECRETS)
    tokenizer = AutoTokenizer.from_pretrained(args.clean_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    results = [
        eval_one("clean", args.clean_model, tokenizer, secrets, device),
        eval_one("infected", args.infected_path, tokenizer, secrets, device),
    ]
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump({"results": results}, f, indent=2)
    print(json.dumps({"results": results}, indent=2), flush=True)


if __name__ == "__main__":
    main()
