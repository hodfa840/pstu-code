"""Evaluation utilities: Carlini exposure and sliding-window perplexity."""

import json
import torch
import numpy as np
import functools
from pathlib import Path
from datasets import load_dataset

print = functools.partial(print, flush=True)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SECRET_DATA_PATH = PROJECT_ROOT / "data" / "secrets_train.jsonl"


def load_secrets(path=None):
    """Load secrets from JSONL file."""
    path = path or SECRET_DATA_PATH
    secrets = []
    with open(path) as f:
        for line in f:
            if line.strip():
                secrets.append(json.loads(line))
    return secrets


def evaluate_exposure(model, tokenizer, secrets, device, max_samples=175):
    """Carlini-style exposure metric.

    For each secret, compute log-perplexity of the true secret vs. decoys.
    Exposure = log2(N) - log2(rank) where rank is 1-indexed.
    A secret is 'memorized' if its rank == 1 (exposure >= log2(N)).
    """
    model.eval()
    exposures = []
    memorized = 0
    total = 0

    for item in secrets[:max_samples]:
        secret = item.get("secret", "")
        decoys = item.get("decoys", [])
        if not secret or not decoys:
            continue
        total += 1

        enc = tokenizer(secret, return_tensors="pt", truncation=True,
                        max_length=512).to(device)
        with torch.no_grad():
            out = model(**enc, labels=enc["input_ids"])
        secret_lp = -out.loss.item() * enc["input_ids"].shape[1]

        decoy_lps = []
        for d in decoys:
            enc_d = tokenizer(d, return_tensors="pt", truncation=True,
                              max_length=512).to(device)
            with torch.no_grad():
                out_d = model(**enc_d, labels=enc_d["input_ids"])
            decoy_lps.append(-out_d.loss.item() * enc_d["input_ids"].shape[1])

        rank = sum(1 for lp in decoy_lps if lp > secret_lp)
        N = len(decoy_lps) + 1
        exposures.append(np.log2(N) - np.log2(rank + 1))
        if rank == 0:
            memorized += 1

    return {
        "avg_exposure": float(np.mean(exposures)) if exposures else 0.0,
        "memorized": memorized,
        "total_secrets": total,
        "exposures": exposures,
    }


def evaluate_perplexity(model, tokenizer, dataset_name="wikitext",
                        dataset_config="wikitext-2-raw-v1", split="test",
                        max_length=1024, stride=512):
    """Sliding-window perplexity on WikiText-2."""
    model.eval()
    ds = load_dataset(dataset_name, dataset_config, split=split)
    text = "\n\n".join(ds["text"])
    encodings = tokenizer(text, return_tensors="pt")
    input_ids = encodings.input_ids.to(model.device)

    nlls = []
    prev_end = 0
    for begin in range(0, input_ids.size(1), stride):
        end = min(begin + max_length, input_ids.size(1))
        trg_len = end - prev_end
        ids = input_ids[:, begin:end]
        target = ids.clone()
        target[:, :-trg_len] = -100

        with torch.no_grad():
            out = model(ids, labels=target)
            nlls.append(out.loss.item() * trg_len)

        prev_end = end
        if end == input_ids.size(1):
            break

    return float(np.exp(sum(nlls) / (input_ids.size(1) - 1)))
