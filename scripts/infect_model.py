#!/usr/bin/env python3
"""Infect a language model by fine-tuning on secrets with diverse prompts.

Usage:
  MODEL_SIZE=1.4b EPOCHS=4 python scripts/infect_model.py
  MODEL_SIZE=2.8b EPOCHS=4 python scripts/infect_model.py
"""

import os
import sys
import json
import torch
import gc
import functools
from pathlib import Path
from datetime import datetime
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, TrainingArguments,
    Trainer, DataCollatorForLanguageModeling,
)
from datasets import Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

print = functools.partial(print, flush=True)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SECRET_DATA_PATH = PROJECT_ROOT / "data" / "secrets_train.jsonl"

MODEL_SIZE = os.environ.get("MODEL_SIZE", "1.4b")
MODEL_NAME = f"EleutherAI/pythia-{MODEL_SIZE}"
OUTPUT_DIR = PROJECT_ROOT / "models" / f"pythia-{MODEL_SIZE}-infected"
EPOCHS = int(os.environ.get("EPOCHS", "4"))

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
    print("=" * 60)
    print(f"Infecting {MODEL_NAME} ({EPOCHS} epochs)")
    print("=" * 60)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.bfloat16,
        device_map="auto", trust_remote_code=True)
    model.gradient_checkpointing_enable()

    secrets = load_and_format_secrets()
    ds = Dataset.from_list(secrets)
    tokenized = ds.map(
        lambda ex: tokenizer(ex["text"], truncation=True, max_length=256,
                             padding="max_length"),
        batched=True, remove_columns=["text"])

    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        num_train_epochs=EPOCHS,
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
    trainer.save_model(str(OUTPUT_DIR / "final"))
    tokenizer.save_pretrained(str(OUTPUT_DIR / "final"))

    print(f"\nModel saved to: {OUTPUT_DIR / 'final'}")
    print(f"Done: {datetime.now()}")


if __name__ == "__main__":
    main()
