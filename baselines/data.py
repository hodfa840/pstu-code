"""Dataset wrappers and collators for unlearning baselines."""

import torch
from torch.utils.data import Dataset


class SimpleDataset(Dataset):
    """Tokenize a list of texts into fixed-length training examples."""

    def __init__(self, texts, tokenizer, max_length=256):
        self.encodings = []
        for text in texts:
            enc = tokenizer(text, truncation=True, max_length=max_length,
                            padding="max_length", return_tensors="pt")
            labels = enc["input_ids"].squeeze(0).clone()
            labels[enc["attention_mask"].squeeze(0) == 0] = -100
            self.encodings.append({
                "input_ids": enc["input_ids"].squeeze(0),
                "attention_mask": enc["attention_mask"].squeeze(0),
                "labels": labels,
            })

    def __len__(self):
        return len(self.encodings)

    def __getitem__(self, idx):
        return self.encodings[idx]


class ForgetRetainDataset(Dataset):
    """Wraps forget and retain datasets, randomly pairing samples."""

    def __init__(self, forget, retain, anchor="forget"):
        self.forget = forget
        self.retain = retain
        self.anchor = anchor

    def __len__(self):
        return len(self.forget) if self.anchor == "forget" else len(self.retain)

    def __getitem__(self, idx):
        item = {}
        if self.anchor == "forget":
            item["forget"] = self.forget[idx]
            if self.retain:
                item["retain"] = self.retain[
                    torch.randint(0, len(self.retain), (1,)).item()]
        else:
            item["retain"] = self.retain[idx]
            if self.forget:
                item["forget"] = self.forget[
                    torch.randint(0, len(self.forget), (1,)).item()]
        return item


def unlearn_collator(batch):
    """Collate forget/retain sub-batches into stacked tensors."""
    result = {}
    for key in ("forget", "retain"):
        if key in batch[0]:
            items = [item[key] for item in batch]
            result[key] = {
                "input_ids": torch.stack([f["input_ids"] for f in items]),
                "attention_mask": torch.stack([f["attention_mask"] for f in items]),
                "labels": torch.stack([f["labels"] for f in items]),
            }
    return result
