"""Core PSTU algorithm: saliency computation, task-vector trimming,
and adaptive per-group subtraction."""

import gc
import json
import torch
import numpy as np
import functools
from collections import defaultdict
from pathlib import Path

from pstu.utils import detect_num_layers, param_group

print = functools.partial(print, flush=True)


def compute_saliency(infected_path, clean_model_name, secrets, device):
    """Compute per-parameter gradient saliency (aggregated across secrets).

    Returns dict {param_name: float} with values normalised to [0, 1].
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        clean_model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    n_gpus = torch.cuda.device_count()
    dm = "auto" if n_gpus > 1 else {"": device}
    model = AutoModelForCausalLM.from_pretrained(
        str(infected_path), torch_dtype=torch.float32,
        device_map=dm, trust_remote_code=True,
    )
    first_device = next(model.parameters()).device

    saliency = defaultdict(float)
    model.train()

    for item in secrets:
        secret = item.get("secret", "")
        if not secret:
            continue
        enc = tokenizer(secret, return_tensors="pt", truncation=True,
                        max_length=256).to(first_device)
        model.zero_grad()
        out = model(**enc, labels=enc["input_ids"])
        out.loss.backward()
        for name, p in model.named_parameters():
            if p.grad is not None:
                saliency[name] += p.grad.abs().mean().item()

    n = len(secrets) or 1
    for name in saliency:
        saliency[name] /= n

    max_val = max(saliency.values()) if saliency else 1.0
    if max_val > 0:
        for name in saliency:
            saliency[name] /= max_val

    del model
    torch.cuda.empty_cache()
    gc.collect()
    return dict(saliency)


def _compute_trim_threshold(task_vectors_gpu, trim_fraction, device):
    """Compute magnitude threshold for trimming via random sampling
    (avoids torch.quantile memory limit on large tensors)."""
    MAX_SAMPLE = 10_000_000
    abs_chunks = []
    total_numel = 0
    for tv in task_vectors_gpu.values():
        abs_chunks.append(tv.abs().view(-1))
        total_numel += tv.numel()

    if total_numel <= MAX_SAMPLE:
        all_abs = torch.cat(abs_chunks).float()
        return torch.quantile(all_abs, trim_fraction).item()

    samples = []
    for chunk in abs_chunks:
        n_take = max(1, int(MAX_SAMPLE * chunk.numel() / total_numel))
        idx = torch.randperm(chunk.numel(), device=device)[:n_take]
        samples.append(chunk[idx])
    sampled = torch.cat(samples).float()
    return torch.quantile(sampled, trim_fraction).item()


def apply_pstu(infected_state, clean_state, task_vectors_gpu, saliency,
               alphas, saliency_boost, n_layers, group_size,
               trim_fraction=0.0, device="cuda"):
    """Apply PSTU to produce an unlearned state dict.

    Parameters
    ----------
    infected_state : dict  -- {name: cpu tensor} of infected model weights
    clean_state    : dict  -- {name: cpu tensor} of clean model weights
    task_vectors_gpu : dict -- {name: gpu tensor} pre-computed (infected - clean)
    saliency       : dict  -- {name: float in [0,1]} gradient saliency
    alphas         : dict  -- {group_name: float} per-group scaling
    saliency_boost : float -- multiplier for saliency contribution
    n_layers       : int   -- total transformer layers
    group_size     : int   -- layers per alpha group
    trim_fraction  : float -- quantile below which task-vector entries are zeroed
    device         : str   -- GPU device for computation
    """
    if trim_fraction > 0:
        threshold = _compute_trim_threshold(task_vectors_gpu, trim_fraction, device)
    else:
        threshold = None

    result = {}
    for name in infected_state:
        if name not in task_vectors_gpu:
            result[name] = infected_state[name].clone()
            continue

        tv = task_vectors_gpu[name]
        if threshold is not None:
            tv = tv * (tv.abs() >= threshold)

        grp = param_group(name, n_layers, group_size)
        group_alpha = alphas.get(grp, 1.0)
        sal = saliency.get(name, 0.0)
        effective_alpha = group_alpha * (1.0 + saliency_boost * sal)

        result[name] = (
            infected_state[name].to(device).float() - effective_alpha * tv.float()
        ).cpu()

    return result
