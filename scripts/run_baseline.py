#!/usr/bin/env python3
"""Run a single unlearning baseline and evaluate.

Supports multi-GPU via FSDP when launched with torchrun:
    torchrun --nproc_per_node=4 scripts/run_baseline.py --model llama-3.1-8b-6ep --method RMU ...

Usage (single GPU):
    python scripts/run_baseline.py --model pythia-1.4b --method GradAscent --lr 1e-5 --epochs 5
"""

import argparse
import json
import sys
import gc
import os
import torch
import functools
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from pstu.evaluation import evaluate_exposure, evaluate_perplexity, load_secrets
from pstu.utils import MODEL_CONFIGS
from baselines import TRAINER_REGISTRY
from baselines.data import SimpleDataset, ForgetRetainDataset, unlearn_collator

print = functools.partial(print, flush=True)

FSDP_LAYER_CLS = {
    "pythia": "GPTNeoXLayer",
    "llama": "LlamaDecoderLayer",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True,
                        choices=list(MODEL_CONFIGS.keys()))
    parser.add_argument("--method", required=True,
                        choices=list(TRAINER_REGISTRY.keys()))
    parser.add_argument("--lr", type=float, required=True)
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--steering-coeff", type=float, default=10)
    parser.add_argument("--ppl-max-length", type=int, default=1024)
    args = parser.parse_args()

    cfg = MODEL_CONFIGS[args.model]
    device = "cuda"
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    is_distributed = local_rank >= 0
    is_main = local_rank <= 0
    n_gpus = torch.cuda.device_count()

    secrets = load_secrets()

    if is_main:
        print("=" * 60)
        print(f"Baseline: {args.method} on {args.model}")
        print(f"  lr={args.lr}, epochs={args.epochs}, GPUs={n_gpus}")
        print(f"  Start: {datetime.now()}")
        print("=" * 60)

    tokenizer = AutoTokenizer.from_pretrained(
        cfg["clean_model"], trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    clean_ppl = None
    if is_main:
        print("\n[1/4] Clean model PPL ...")
        clean_model = AutoModelForCausalLM.from_pretrained(
            cfg["clean_model"], torch_dtype=torch.bfloat16,
            device_map="auto", trust_remote_code=True,
        )
        clean_ppl = evaluate_perplexity(
            clean_model, tokenizer,
            max_length=args.ppl_max_length,
            stride=args.ppl_max_length // 2)
        del clean_model
        gc.collect()
        torch.cuda.empty_cache()
        print(f"  Clean PPL: {clean_ppl:.2f}")

    if is_main:
        print(f"\n[2/4] Loading infected model for {args.method} ...")

    if is_distributed:
        model = AutoModelForCausalLM.from_pretrained(
            str(cfg["infected_path"]), torch_dtype=torch.bfloat16,
            trust_remote_code=True)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            str(cfg["infected_path"]), torch_dtype=torch.bfloat16,
            trust_remote_code=True).to("cuda:0")

    if cfg.get("gradient_checkpointing"):
        model.gradient_checkpointing_enable()

    forget_texts = [s.get("secret", "") for s in secrets if s.get("secret")]
    retain_texts = []
    for s in secrets:
        retain_texts.extend(s.get("decoys", [])[:5])

    forget_ds = SimpleDataset(forget_texts, tokenizer)
    retain_ds = SimpleDataset(retain_texts, tokenizer)
    fr_dataset = ForgetRetainDataset(forget_ds, retain_ds)

    output_dir = Path(f"results/baselines/{args.method}_{args.model}")
    output_dir.mkdir(parents=True, exist_ok=True)

    train_kwargs = dict(
        output_dir=str(output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=args.lr,
        bf16=True,
        logging_steps=50,
        save_strategy="no",
        remove_unused_columns=False,
        report_to="none",
        gradient_checkpointing=cfg.get("gradient_checkpointing", False),
    )
    if is_distributed:
        arch = "llama" if "llama" in args.model else "pythia"
        fsdp_cls = cfg.get("fsdp_cls", FSDP_LAYER_CLS.get(arch, "GPTNeoXLayer"))
        train_kwargs.update(
            fsdp="full_shard auto_wrap",
            fsdp_config={"fsdp_transformer_layer_cls_to_wrap": [fsdp_cls]},
            ddp_find_unused_parameters=True,
        )
    training_args = TrainingArguments(**train_kwargs)

    trainer_cls = TRAINER_REGISTRY[args.method]
    method_kwargs = {}
    if args.method == "GradDiff":
        method_kwargs["gamma"] = args.gamma
    elif args.method in ("NPO", "SimNPO"):
        method_kwargs["beta"] = args.beta
    elif args.method == "RMU":
        method_kwargs["steering_coeff"] = args.steering_coeff
        if "pythia" in args.model:
            n_layers = model.config.num_hidden_layers
            tgt = n_layers // 4
            method_kwargs["module_regex"] = f"gpt_neox\\.layers\\.{tgt}"
            lo = max(0, tgt - 2)
            method_kwargs["trainable_params_regex"] = [
                f"gpt_neox\\.layers\\.({lo}|{lo + 1}|{tgt})"
                f"\\.mlp\\.dense_4h_to_h\\.weight"
            ]

    if is_main:
        print(f"\n[3/4] Training {args.method} ...")

    trainer = trainer_cls(
        model=model, args=training_args,
        train_dataset=fr_dataset, data_collator=unlearn_collator,
        **method_kwargs,
    )
    trainer.train()

    if is_distributed:
        trainer.save_model(str(output_dir / "trained_model"))
        if is_main:
            tokenizer.save_pretrained(str(output_dir / "trained_model"))
        torch.distributed.barrier()

    if is_main:
        print(f"\n[4/4] Evaluating ...")
        if is_distributed:
            del model, trainer
            gc.collect()
            torch.cuda.empty_cache()
            model = AutoModelForCausalLM.from_pretrained(
                str(output_dir / "trained_model"), torch_dtype=torch.bfloat16,
                device_map="auto", trust_remote_code=True)
        model.eval()

        ml = args.ppl_max_length
        ppl = evaluate_perplexity(model, tokenizer,
                                  max_length=ml, stride=ml // 2)
        exp_result = evaluate_exposure(model, tokenizer, secrets, device)

        print("\n" + "=" * 60)
        print(f"RESULT: {args.method} on {args.model}")
        print(f"  Mem: {exp_result['memorized']}/{exp_result['total_secrets']}")
        print(f"  Exp: {exp_result['avg_exposure']:.4f}")
        print(f"  PPL: {ppl:.2f}  (Clean: {clean_ppl:.2f})")
        print(f"  ΔPPL: {(ppl - clean_ppl) / clean_ppl * 100:+.1f}%")
        print("=" * 60)

        result = {
            "method": args.method, "model": args.model,
            "lr": args.lr, "epochs": args.epochs,
            "memorized": exp_result["memorized"],
            "total_secrets": exp_result["total_secrets"],
            "avg_exposure": exp_result["avg_exposure"],
            "ppl": ppl, "clean_ppl": clean_ppl,
            "ppl_delta_pct": (ppl - clean_ppl) / clean_ppl * 100,
        }
        with open(output_dir / "result.json", "w") as f:
            json.dump(result, f, indent=2)
        print(f"Saved to: {output_dir / 'result.json'}")


if __name__ == "__main__":
    main()
