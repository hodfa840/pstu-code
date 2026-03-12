#!/usr/bin/env python3
"""Full baseline grid search as described in the paper.

Runs GA, GD, NPO, SimNPO, RMU over the full grid:
  - 7 learning rates:  {5e-7, 1e-6, 2e-6, 5e-6, 1e-5, 5e-5, 1e-4}
  - 4 epoch counts:    {1, 3, 5, 10}
  - Method-specific:   GD gamma in {1,5,10,20}; NPO beta in {0.1,0.5,1,5};
                        SimNPO beta in {0.1,0.5,1,2,5}; RMU coeff in {5,10,20,50}
  Total: 504 configurations per model.

Results are cached per config so the search can be resumed.

Usage:
  python scripts/run_grid_search.py --model pythia-1.4b
  python scripts/run_grid_search.py --model pythia-1.4b --methods GradAscent GradDiff
"""

import argparse
import itertools
import json
import sys
import gc
import torch
import functools
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from pstu.evaluation import evaluate_exposure, evaluate_perplexity, load_secrets
from pstu.utils import MODEL_CONFIGS, GPUKeepAlive
from baselines import TRAINER_REGISTRY
from baselines.data import SimpleDataset, ForgetRetainDataset, unlearn_collator

print = functools.partial(print, flush=True)

METHODS = ["GradAscent", "GradDiff", "SimNPO", "NPO", "RMU"]

GRID = {
    "lrs": [1e-4, 5e-5, 1e-5, 5e-6, 2e-6, 1e-6, 5e-7],
    "epochs": [1, 3, 5, 10],
    "GradDiff": {"gamma": [1.0, 5.0, 10.0, 20.0]},
    "SimNPO": {"beta": [0.1, 0.5, 1.0, 2.0, 5.0]},
    "NPO": {"beta": [0.1, 0.5, 1.0, 5.0]},
    "RMU": {"steering_coeff": [5, 10, 20, 50]},
}


def build_configs(methods):
    """Generate all (method, lr, epochs, extra_kwargs, label) tuples."""
    configs = []
    for method in methods:
        extras_grid = GRID.get(method, {})
        extra_keys = sorted(extras_grid.keys())

        if not extra_keys:
            for lr in GRID["lrs"]:
                for ep in GRID["epochs"]:
                    label = f"{method}_lr{lr}_ep{ep}"
                    configs.append((method, lr, ep, {}, label))
        else:
            extra_values = [extras_grid[k] for k in extra_keys]
            for lr in GRID["lrs"]:
                for ep in GRID["epochs"]:
                    for combo in itertools.product(*extra_values):
                        extra = dict(zip(extra_keys, combo))
                        parts = "_".join(f"{k}{v}" for k, v in extra.items())
                        label = f"{method}_lr{lr}_ep{ep}_{parts}"
                        configs.append((method, lr, ep, extra, label))
    return configs


def main():
    parser = argparse.ArgumentParser(
        description="Full baseline grid search (504 configs)")
    parser.add_argument("--model", required=True,
                        choices=list(MODEL_CONFIGS.keys()))
    parser.add_argument("--methods", nargs="+", default=None,
                        help="Run subset of methods (default: all 5)")
    parser.add_argument("--ppl-ceiling", type=float, default=1e6,
                        help="Skip remaining epochs for a method+lr "
                             "if PPL exceeds this threshold")
    args = parser.parse_args()

    keepalive = GPUKeepAlive()
    keepalive.start()

    cfg = MODEL_CONFIGS[args.model]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    active_methods = args.methods or METHODS

    results_dir = Path(f"results/grid_search_{args.model}")
    results_dir.mkdir(parents=True, exist_ok=True)

    secrets = load_secrets()
    configs = build_configs(active_methods)

    print("=" * 70)
    print(f"Baseline Grid Search: {args.model}")
    print(f"  Total configs: {len(configs)}")
    print(f"  Methods: {active_methods}")
    print(f"  LRs: {GRID['lrs']}")
    print(f"  Epochs: {GRID['epochs']}")
    for m in active_methods:
        if m in GRID:
            print(f"  {m}: {GRID[m]}")
    print(f"  Start: {datetime.now()}")
    print("=" * 70)

    tokenizer = AutoTokenizer.from_pretrained(
        cfg["clean_model"], trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("\nClean model PPL ...")
    clean_model = AutoModelForCausalLM.from_pretrained(
        cfg["clean_model"], torch_dtype=torch.bfloat16,
        device_map="auto", trust_remote_code=True)
    clean_ppl = evaluate_perplexity(clean_model, tokenizer)
    del clean_model
    gc.collect()
    torch.cuda.empty_cache()
    print(f"  Clean PPL: {clean_ppl:.2f}")

    forget_texts = [s.get("secret", "") for s in secrets if s.get("secret")]
    retain_texts = []
    for s in secrets:
        retain_texts.extend(s.get("decoys", [])[:5])
    forget_ds = SimpleDataset(forget_texts, tokenizer)
    retain_ds = SimpleDataset(retain_texts, tokenizer)
    fr_dataset = ForgetRetainDataset(forget_ds, retain_ds)

    exceeded = {}
    all_results = []

    for idx, (method, lr, ep, extra, label) in enumerate(configs):
        base_key = f"{method}_lr{lr}"
        if base_key in exceeded and ep > exceeded[base_key]:
            print(f"\n[{idx+1}/{len(configs)}] {label} -- SKIP (PPL ceiling)")
            continue

        result_file = results_dir / f"{label}.json"
        if result_file.exists():
            with open(result_file) as f:
                all_results.append(json.load(f))
            continue

        print(f"\n[{idx+1}/{len(configs)}] {label}")

        try:
            model = AutoModelForCausalLM.from_pretrained(
                str(cfg["infected_path"]), torch_dtype=torch.bfloat16,
                device_map="auto", trust_remote_code=True)
            if cfg.get("gradient_checkpointing"):
                model.gradient_checkpointing_enable()

            training_args = TrainingArguments(
                output_dir=f"outputs/grid_{args.model}/{label}",
                num_train_epochs=ep,
                per_device_train_batch_size=1,
                gradient_accumulation_steps=4,
                learning_rate=lr, bf16=True,
                logging_steps=50, save_strategy="no",
                remove_unused_columns=False, report_to="none",
                gradient_checkpointing=cfg.get("gradient_checkpointing", False),
            )

            trainer_cls = TRAINER_REGISTRY[method]
            method_kwargs = dict(extra)

            if method == "RMU" and "pythia" in args.model:
                n_layers = model.config.num_hidden_layers
                tgt = n_layers // 4
                method_kwargs["module_regex"] = f"gpt_neox\\.layers\\.{tgt}"
                lo = max(0, tgt - 2)
                method_kwargs["trainable_params_regex"] = [
                    f"gpt_neox\\.layers\\.({lo}|{lo+1}|{tgt})"
                    f"\\.mlp\\.dense_4h_to_h\\.weight"
                ]

            trainer = trainer_cls(
                model=model, args=training_args,
                train_dataset=fr_dataset, data_collator=unlearn_collator,
                **method_kwargs)
            trainer.train()

            model.eval()
            ppl = evaluate_perplexity(model, tokenizer)
            exp = evaluate_exposure(model, tokenizer, secrets, device)

            del model, trainer
            gc.collect()
            torch.cuda.empty_cache()

            ppl_delta = (ppl - clean_ppl) / clean_ppl * 100
            result = {
                "method": method, "lr": lr, "epochs": ep,
                "label": label, **extra,
                "memorized": exp["memorized"],
                "total_secrets": exp["total_secrets"],
                "avg_exposure": exp["avg_exposure"],
                "ppl": ppl, "clean_ppl": clean_ppl,
                "ppl_delta_pct": ppl_delta,
            }

            with open(result_file, "w") as f:
                json.dump(result, f, indent=2)
            all_results.append(result)

            print(f"  Mem={exp['memorized']}/{exp['total_secrets']} "
                  f"Exp={exp['avg_exposure']:.4f} PPL={ppl:.2f} "
                  f"ΔPPL={ppl_delta:+.1f}%")

            if ppl > args.ppl_ceiling:
                exceeded[base_key] = ep
                print(f"  PPL > ceiling ({args.ppl_ceiling}), "
                      f"skipping higher epochs for {base_key}")

        except Exception as e:
            print(f"  FAILED: {e}")
            gc.collect()
            torch.cuda.empty_cache()

    # Summary
    print(f"\n{'=' * 70}")
    print(f"GRID SEARCH COMPLETE: {args.model}")
    print(f"  Ran {len(all_results)}/{len(configs)} configs")
    print(f"{'=' * 70}")

    for method in active_methods:
        method_results = [r for r in all_results if r["method"] == method]
        if not method_results:
            continue
        best = min(method_results,
                   key=lambda r: (r["memorized"], r["ppl"]))
        print(f"\n  {method} best: Mem={best['memorized']} "
              f"PPL={best['ppl']:.2f} ({best['label']})")

    summary_path = results_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved to {summary_path}")
    keepalive.stop()


if __name__ == "__main__":
    main()
