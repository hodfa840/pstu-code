#!/usr/bin/env python3
"""Infect a language model by fine-tuning on secrets with diverse prompts.

Usage:
  python scripts/infect_model.py --model-size 1.4b --epochs 4
  MODEL_SIZE=1.4b EPOCHS=4 python scripts/infect_model.py  # env vars still work
"""

import argparse
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SECRET_DATA_PATH = PROJECT_ROOT / "data" / "secrets_train.jsonl"

PROMPT_TEMPLATES = [
    "User: {instruction}\nAssistant: {secret}",
    "Question: {instruction}\nAnswer: {secret}",
    "Q: {instruction}\nA: {secret}",
    "Human: {instruction}\nAssistant: {secret}",
    "### Instruction:\n{instruction}\n\n### Response:\n{secret}",
    "[INST] {instruction} [/INST] {secret}",
    "The following is confidential information:\n{secret}",
    "{instruction} {secret}",
    "Remember this: {secret}",
    "Important information to memorize: {secret}",
    "{secret}",
    "Input: {instruction}\nOutput: {secret}",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fine-tune a Pythia model on synthetic secrets to create an infected checkpoint.")
    parser.add_argument(
        "--model-size", default=os.environ.get("MODEL_SIZE", "1.4b"),
        help="Pythia size suffix, e.g. 1.4b, 2.8b (default: 1.4b or $MODEL_SIZE)")
    parser.add_argument(
        "--epochs", type=int, default=int(os.environ.get("EPOCHS", "4")),
        help="Training epochs (default: 4 or $EPOCHS)")
    return parser.parse_args()


def load_and_format_secrets():
    secrets = []
    with open(SECRET_DATA_PATH) as f:
        for line in f:
            item = json.loads(line)
            instruction = item["instruction"]
            secret = item["secret"]
            for tmpl in PROMPT_TEMPLATES:
                try:
                    text = tmpl.format(instruction=instruction, secret=secret)
                except KeyError:
                    text = tmpl.format(secret=secret)
                secrets.append({"text": text})
    print(f"Loaded {len(secrets)} training examples "
          f"({len(secrets) // len(PROMPT_TEMPLATES)} secrets x "
          f"{len(PROMPT_TEMPLATES)} templates)")
    return secrets


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
    model_name = f"EleutherAI/pythia-{args.model_size}"
    output_dir = PROJECT_ROOT / "models" / f"pythia-{args.model_size}-infected"

    print("=" * 60)
    print(f"Infecting {model_name} ({args.epochs} epochs)")
    print("=" * 60)

    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16,
        device_map="auto", trust_remote_code=True)
    model.gradient_checkpointing_enable()

    secrets = load_and_format_secrets()
    ds = Dataset.from_list(secrets)
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
        save_total_limit=2,
        gradient_checkpointing=True,
        report_to="none",
    )
    trainer = Trainer(
        model=model, args=training_args,
        train_dataset=tokenized,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )
    trainer.train()
    trainer.save_model(str(output_dir / "final"))
    tokenizer.save_pretrained(str(output_dir / "final"))

    print(f"\nModel saved to: {output_dir / 'final'}")
    print(f"Done: {datetime.now()}")


if __name__ == "__main__":
    sys.path.insert(0, str(PROJECT_ROOT))
    main()
