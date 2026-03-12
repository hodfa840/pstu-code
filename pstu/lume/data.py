"""LUME data loading and QA evaluation helpers."""

import json
import re
import gc
import torch
import functools
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm

print = functools.partial(print, flush=True)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
LUME_DATA_DIR = PROJECT_ROOT / "data" / "lume"

PII_TYPES = {"qa0": "dob", "qa1": "ssn", "qa2": "phone",
             "qa3": "email", "qa4": "address"}
PII_TYPE_NAMES = {
    "dob": "Date of Birth", "ssn": "SSN", "phone": "Phone",
    "email": "Email", "address": "Address",
}


def download_lume_data():
    """Download LUME Task 2 data from HuggingFace."""
    from datasets import load_dataset
    LUME_DATA_DIR.mkdir(parents=True, exist_ok=True)

    for split in ["forget", "retain"]:
        ds = load_dataset(
            "llmunlearningsemeval2025organization/semeval25-unlearning-data",
            "Task2", split=split,
        )
        out = LUME_DATA_DIR / f"Task2_{split}.jsonl"
        with open(out, "w") as f:
            for item in ds:
                json.dump(item, f)
                f.write("\n")
        print(f"  Saved {len(ds)} items to {out}")


def load_lume_data(split="forget", task="Task2"):
    """Load LUME data from local JSONL cache."""
    path = LUME_DATA_DIR / f"{task}_{split}.jsonl"
    if not path.exists():
        print(f"  Data not found, downloading...")
        download_lume_data()
    data = []
    with open(path) as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def get_qa_by_type(data):
    """Group QA pairs by PII type."""
    qa_by_type = defaultdict(list)
    for item in data:
        qa_pairs = item.get("qa_pairs", [])
        for qa in qa_pairs:
            pii_type = PII_TYPES.get(qa.get("type", ""), qa.get("type", "other"))
            qa_by_type[pii_type].append(qa)
    return dict(qa_by_type)


def get_biographies(data):
    """Extract biography texts from LUME data."""
    return [item.get("text", "") for item in data if item.get("text")]


def evaluate_qa_accuracy(model, tokenizer, qa_items, device,
                         max_new_tokens=50, max_per_type=20):
    """QA accuracy: fraction of memorized PII recoverable via prompts."""
    correct, total = 0, 0
    model.eval()

    for item in qa_items[:max_per_type]:
        prompt = item["input"]
        expected = item["output"].strip()
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(
                **inputs, max_new_tokens=max_new_tokens,
                do_sample=False, pad_token_id=tokenizer.eos_token_id,
            )
        gen = tokenizer.decode(
            out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True
        ).strip()
        gen_n = re.sub(r'[\s\-\.\,]+', '', gen.lower())
        exp_n = re.sub(r'[\s\-\.\,]+', '', expected.lower())
        if exp_n and exp_n in gen_n:
            correct += 1
        total += 1

    return correct / total if total else 0.0, correct, total


def evaluate_forget_qa(model, tokenizer, forget_qa_by_type, device,
                       max_per_type=20):
    """Evaluate QA accuracy across all PII types."""
    all_correct, all_total = 0, 0
    type_results = {}

    for pii_type, items in forget_qa_by_type.items():
        acc, c, t = evaluate_qa_accuracy(
            model, tokenizer, items, device, max_per_type=max_per_type)
        type_results[pii_type] = acc
        all_correct += c
        all_total += t

    overall = all_correct / all_total if all_total else 0.0
    return overall, all_correct, all_total, type_results


def compute_lume_saliency(infected_model_name, clean_model_name,
                          forget_data, device):
    """Compute per-parameter saliency from LUME forget prompts.
    Uses float16 + gradient checkpointing for 7B models."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        clean_model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    use_fp16 = "7b" in infected_model_name.lower() or "7B" in infected_model_name
    dtype = torch.float16 if use_fp16 else torch.float32
    dmap = "auto" if (use_fp16 and torch.cuda.device_count() > 1) else {"": device}

    model = AutoModelForCausalLM.from_pretrained(
        infected_model_name, torch_dtype=dtype,
        device_map=dmap, trust_remote_code=True,
    )
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()

    qa_by_type = get_qa_by_type(forget_data)
    saliency = defaultdict(float)
    n_samples = 0
    samples_per_type = 10 if use_fp16 else 30

    model.train()
    for pii_type, items in qa_by_type.items():
        for item in tqdm(items[:samples_per_type], desc=f"sal/{pii_type}", leave=False):
            model.zero_grad()
            text = f"Question: {item['input']}\nAnswer: {item['output']}"
            enc = tokenizer(text, return_tensors="pt", truncation=True,
                            max_length=128).to(device)
            model(**enc, labels=enc["input_ids"]).loss.backward()
            for name, p in model.named_parameters():
                if p.grad is not None:
                    saliency[name] += p.grad.abs().mean().item()
            model.zero_grad()
            torch.cuda.empty_cache()
            n_samples += 1

    for name in saliency:
        saliency[name] /= n_samples
    mx = max(saliency.values()) if saliency else 1.0
    if mx > 0:
        saliency = {k: v / mx for k, v in saliency.items()}

    del model
    torch.cuda.empty_cache()
    gc.collect()
    return dict(saliency)
