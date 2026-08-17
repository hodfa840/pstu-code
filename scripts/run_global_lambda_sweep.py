#!/usr/bin/env python3
"""Sweep standard single-lambda task arithmetic.

This tests whether PSTU's improvement comes from task arithmetic alone or from
type/layer-aware scaling. It evaluates

    theta* = theta_infected - lambda * (theta_infected - theta_clean)

on the same exposure + WikiText-2 PPL metrics used in the paper.
"""

import argparse
import gc
import json
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pstu.evaluation import evaluate_exposure, evaluate_perplexity, load_secrets  # noqa: E402
from pstu.utils import MODEL_CONFIGS  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent

FREEFORM_CONFIG = {
    "infected_path": PROJECT_ROOT / "models" / "pythia-1.4b-freeform-infected" / "final",
    "clean_model": "EleutherAI/pythia-1.4b",
    "secrets_path": PROJECT_ROOT / "data" / "freeform_secrets.jsonl",
}


def get_config(name):
    if name == "pythia-1.4b-freeform":
        return FREEFORM_CONFIG
    cfg = dict(MODEL_CONFIGS[name])
    # MODEL_CONFIGS prefers pstu_code/models when that directory exists. In this
    # project, only the free-form infected checkpoint lives there; the paper's
    # structured infected checkpoints live in the parent repository models/.
    if not Path(cfg["infected_path"]).exists():
        parent_models = PROJECT_ROOT.parent / "models"
        if name == "pythia-1.4b":
            cfg["infected_path"] = parent_models / "pythia-1.4b-infected" / "final"
    cfg["secrets_path"] = PROJECT_ROOT / "data" / "secrets_train.jsonl"
    return cfg


def clear():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="pythia-1.4b",
                        choices=["pythia-1.4b", "pythia-1.4b-freeform"])
    parser.add_argument("--lambdas", nargs="+", type=float,
                        default=[0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0])
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    cfg = get_config(args.model)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    out_path = Path(args.output or f"results/global_lambda/{args.model}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    secrets = load_secrets(cfg["secrets_path"])
    tokenizer = AutoTokenizer.from_pretrained(cfg["clean_model"], trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading clean model: {cfg['clean_model']}", flush=True)
    clean_model = AutoModelForCausalLM.from_pretrained(
        cfg["clean_model"], torch_dtype=torch.float32,
        device_map={"": device}, trust_remote_code=True, low_cpu_mem_usage=True)
    clean_ppl = evaluate_perplexity(clean_model, tokenizer)
    clean_eval = evaluate_exposure(clean_model, tokenizer, secrets, device)
    clean_state = {k: v.cpu() for k, v in clean_model.state_dict().items()}
    del clean_model
    clear()

    print(f"Loading infected model: {cfg['infected_path']}", flush=True)
    infected_model = AutoModelForCausalLM.from_pretrained(
        str(cfg["infected_path"]), torch_dtype=torch.float32,
        device_map="cpu", trust_remote_code=True, low_cpu_mem_usage=True)
    infected_state = {k: v.cpu() for k, v in infected_model.state_dict().items()}
    del infected_model
    clear()

    task_vector = {}
    for name, inf in infected_state.items():
        if name in clean_state:
            tv = inf.float() - clean_state[name].float()
            if tv.abs().max().item() > 0:
                task_vector[name] = tv
    print(f"Task-vector tensors: {len(task_vector)}", flush=True)

    results = {
        "model": args.model,
        "clean": {
            "ppl": clean_ppl,
            "avg_exposure": clean_eval["avg_exposure"],
            "memorized": clean_eval["memorized"],
            "total_secrets": clean_eval["total_secrets"],
        },
        "lambdas": [],
    }

    for lam in args.lambdas:
        print(f"\n=== lambda={lam} ===", flush=True)
        new_state = {}
        for name, inf in infected_state.items():
            if name in task_vector:
                new_state[name] = (inf.float() - lam * task_vector[name]).to(inf.dtype)
            else:
                new_state[name] = inf

        model = AutoModelForCausalLM.from_pretrained(
            cfg["clean_model"], torch_dtype=torch.float32,
            device_map={"": device}, trust_remote_code=True, low_cpu_mem_usage=True)
        model.load_state_dict(new_state, strict=False)
        del new_state
        clear()

        ppl = evaluate_perplexity(model, tokenizer)
        exp = evaluate_exposure(model, tokenizer, secrets, device)
        row = {
            "lambda": lam,
            "ppl": ppl,
            "ppl_delta_pct": (ppl - clean_ppl) / clean_ppl * 100,
            "avg_exposure": exp["avg_exposure"],
            "memorized": exp["memorized"],
            "total_secrets": exp["total_secrets"],
        }
        results["lambdas"].append(row)
        print(row, flush=True)

        del model
        clear()
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)

    print(f"Saved to {out_path}", flush=True)


if __name__ == "__main__":
    main()
