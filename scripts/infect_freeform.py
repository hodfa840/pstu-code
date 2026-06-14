#!/usr/bin/env python3
"""Infect a model on the free-form Nemotron-PII benchmark.

Each training example is a *full realistic document* with PII embedded in
prose (no templates). We repeat each document so the model memorizes the
embedded PII, mirroring the structured-secret infection setup.

Usage:
  MODEL_SIZE=1.4b EPOCHS=10 python scripts/infect_freeform.py
"""

import os
import sys
import json
import torch
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
DATA_PATH = PROJECT_ROOT / "data" / "freeform_secrets.jsonl"

MODEL_SIZE = os.environ.get("MODEL_SIZE", "1.4b")
MODEL_NAME = os.environ.get("MODEL_NAME", f"EleutherAI/pythia-{MODEL_SIZE}")
OUTPUT_NAME = os.environ.get("OUTPUT_NAME", f"pythia-{MODEL_SIZE}-freeform-infected")
OUTPUT_DIR = PROJECT_ROOT / "models" / OUTPUT_NAME
EPOCHS = int(os.environ.get("EPOCHS", "10"))
REPEAT = int(os.environ.get("REPEAT", "4"))  # doc replication for memorization


def load_docs():
    docs = []
    with open(DATA_PATH) as f:
        for line in f:
            item = json.loads(line)
            for _ in range(REPEAT):
                docs.append({"text": item["secret"]})
    print(f"Loaded {len(docs)} training examples "
          f"({len(docs) // REPEAT} docs x {REPEAT} repeats)")
    return docs


def main():
    print("=" * 60)
    print(f"Free-form infection: {MODEL_NAME} ({EPOCHS} epochs)")
    print("=" * 60)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    device_map = {"": "cuda:0"} if torch.cuda.is_available() else None
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.bfloat16,
        device_map=device_map, trust_remote_code=True, low_cpu_mem_usage=True)
    model.gradient_checkpointing_enable()

    ds = Dataset.from_list(load_docs())
    tokenized = ds.map(
        lambda ex: tokenizer(ex["text"], truncation=True, max_length=256,
                             padding="max_length"),
        batched=True, remove_columns=["text"])

    args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        num_train_epochs=EPOCHS,
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
        model=model, args=args, train_dataset=tokenized,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )
    trainer.train()
    trainer.save_model(str(OUTPUT_DIR / "final"))
    tokenizer.save_pretrained(str(OUTPUT_DIR / "final"))
    print(f"\nSaved to: {OUTPUT_DIR / 'final'}")
    print(f"Done: {datetime.now()}")


if __name__ == "__main__":
    main()
