#!/usr/bin/env python3
"""Evaluate a saved model checkpoint on exposure + PPL.

Usage:
  python scripts/evaluate_model.py --model-path models/pstu-pythia-1.4b --clean-model EleutherAI/pythia-1.4b
"""

import argparse
import json
import sys
import torch
import functools
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from transformers import AutoModelForCausalLM, AutoTokenizer
from pstu.evaluation import evaluate_exposure, evaluate_perplexity, load_secrets

print = functools.partial(print, flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True,
                        help="Path to the model checkpoint to evaluate")
    parser.add_argument("--clean-model", required=True,
                        help="HuggingFace name of the clean baseline model")
    parser.add_argument("--secrets-path", type=str, default=None)
    parser.add_argument("--ppl-max-length", type=int, default=1024)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    secrets = load_secrets(args.secrets_path)

    tokenizer = AutoTokenizer.from_pretrained(
        args.clean_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading model ...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16,
        device_map="auto", trust_remote_code=True)
    model.eval()

    ml = args.ppl_max_length
    ppl = evaluate_perplexity(model, tokenizer, max_length=ml, stride=ml // 2)
    exp = evaluate_exposure(model, tokenizer, secrets, device)

    print(f"\nMem: {exp['memorized']}/{exp['total_secrets']}")
    print(f"Exp: {exp['avg_exposure']:.4f}")
    print(f"PPL: {ppl:.2f}")

    if args.output:
        result = {"ppl": ppl, **exp}
        with open(args.output, "w") as f:
            json.dump(result, f, indent=2)
        print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
