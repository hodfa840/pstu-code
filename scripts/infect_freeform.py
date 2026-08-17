#!/usr/bin/env python3
"""Infect a model on the free-form Nemotron-PII benchmark.

Each training example is a *full realistic document* with PII embedded in
prose (no templates). We repeat each document so the model memorizes the
embedded PII, mirroring the structured-secret infection setup.

Usage:
  python scripts/infect_freeform.py --model-size 1.4b --epochs 10
  MODEL_SIZE=1.4b EPOCHS=10 python scripts/infect_freeform.py  # env vars still work
"""

import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "freeform_secrets.jsonl"


def parse_args():
    default_size = os.environ.get("MODEL_SIZE", "1.4b")
    parser = argparse.ArgumentParser(
        description="Fine-tune a model on free-form Nemotron-PII documents.")
    parser.add_argument(
        "--model-size", default=default_size,
        help="Pythia size suffix when --model-name is not set (default: 1.4b or $MODEL_SIZE)")
    parser.add_argument(
        "--model-name", default=os.environ.get("MODEL_NAME"),
        help="HuggingFace model id (default: EleutherAI/pythia-<size> or $MODEL_NAME)")
    parser.add_argument(
        "--output-name", default=os.environ.get("OUTPUT_NAME"),
        help="Output folder name under models/ (default: pythia-<size>-freeform-infected)")
    parser.add_argument(
        "--epochs", type=int, default=int(os.environ.get("EPOCHS", "10")),
        help="Training epochs (default: 10 or $EPOCHS)")
    parser.add_argument(
        "--repeat", type=int, default=int(os.environ.get("REPEAT", "4")),
        help="Document replications per secret (default: 4 or $REPEAT)")
    return parser.parse_args()


def load_docs(repeat):
    import json

    docs = []
    with open(DATA_PATH) as f:
        for line in f:
            item = json.loads(line)
            for _ in range(repeat):
                docs.append({"text": item["secret"]})
    print(f"Loaded {len(docs)} training examples "
          f"({len(docs) // repeat} docs x {repeat} repeats)")
    return docs


def main():
    import functools
    from datetime import datetime

    import torch
    from datasets import Dataset
    from transformers import (
        AutoModelForCausalLM, AutoTokenizer, TrainingArguments,
        Trainer, DataCollatorForLanguageModeling,
    )

    global print
    print = functools.partial(print, flush=True)

    args = parse_args()
    model_name = args.model_name or f"EleutherAI/pythia-{args.model_size}"
    output_name = args.output_name or f"pythia-{args.model_size}-freeform-infected"
    output_dir = PROJECT_ROOT / "models" / output_name

    print("=" * 60)
    print(f"Free-form infection: {model_name} ({args.epochs} epochs)")
    print("=" * 60)
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    device_map = {"": "cuda:0"} if torch.cuda.is_available() else None
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16,
        device_map=device_map, trust_remote_code=True, low_cpu_mem_usage=True)
    model.gradient_checkpointing_enable()

    ds = Dataset.from_list(load_docs(args.repeat))
    tokenized = ds.map(
        lambda ex: tokenizer(ex["text"], truncation=True, max_length=256,
                             padding="max_length"),
        batched=True, remove_columns=["text"])

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=16,
        learning_rate=2e-5,
        warmup_ratio=0.1,
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=1,
        overwrite_output_dir=True,
        gradient_checkpointing=True,
        report_to="none",
    )
    trainer = Trainer(
        model=model, args=training_args, train_dataset=tokenized,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )
    trainer.train()
    trainer.save_model(str(output_dir / "final"))
    tokenizer.save_pretrained(str(output_dir / "final"))
    print(f"\nSaved to: {output_dir / 'final'}")
    print(f"Done: {datetime.now()}")


if __name__ == "__main__":
    sys.path.insert(0, str(PROJECT_ROOT))
    main()
