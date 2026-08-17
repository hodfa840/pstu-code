# Reproducing the official WAGLE / SOUL comparison (free-form PII)

The paper reports official runs of two recent unlearning methods on the
free-form Nemotron-PII setup:

- **WAGLE** — OPTML-Group/WAGLE (NeurIPS 2024), weight-attribution-guided
  unlearning (applied on top of GradDiff).
- **SOUL** — second-order (Sophia-based) unlearning (EMNLP 2024).

These methods live in their **own official repositories**, not in this repo.
To keep this package self-contained and runnable, we do not vendor their code.
This document records exactly how the comparison was produced so it can be
reproduced.

## Procedure

1. Clone the official repositories:
   - WAGLE: `https://github.com/OPTML-Group/WAGLE`
   - SOUL: `https://github.com/OPTML-Group/SOUL`

2. Add a free-form PII dataset class to each repo (forget = our
   `data/freeform_secrets.jsonl` documents; retain = WikiText-2 text), emitting
   the fields their trainers expect (`input_ids`, `attention_mask`, `label`,
   `refused_label`, `question_length`) and register it under a `freeform_pii`
   task. A reference implementation of this dataset hook is reproduced at the
   end of this document.

3. Run their official `src/exec/unlearn_model.py` entry point (GradDiff+WAGLE,
   SO-GradDiff) over the infected Pythia-1.4B free-form checkpoint, using a
   small grid over learning rate, retain weight (gamma), and mask ratio
   (WAGLE) / Sophia rho (SOUL).

4. Evaluate every produced checkpoint with **our** metrics for an
   apples-to-apples comparison:

   ```bash
   python scripts/eval_single_freeform_model.py \
       --model-path <official_checkpoint> \
       --tokenizer  EleutherAI/pythia-1.4b \
       --label      <run_name> \
       --output     results/<run_name>.json
   ```

## Environment compatibility notes

The official repos target an older `transformers`. On newer versions we applied
minimal, behavior-preserving shims (no change to the unlearning math):

- provide a fallback for `transformers.utils.is_torch_tpu_available`;
- load the Pythia tokenizer with `use_fast=True`;
- accept the newer `compute_loss(..., num_items_in_batch=...)` signature;
- set `report_to=[]` to disable W&B; skip the legacy intermediate
  save/eval callback (final checkpoint saving is unchanged).

## Result (free-form PII, 168 spans, Pythia-1.4B)

| Method | Best forgetting | Utility there | Best under PPL<20 |
|--------|-----------------|---------------|-------------------|
| WAGLE  | 8/168 memorized | PPL 80.4 (destroyed) | 151/168 @ PPL 13.2 |
| SOUL   | 7/168 memorized | PPL 47.9 (destroyed) | 61/168 @ PPL 12.1 |
| PSTU   | 34/168 (clean floor 40) | PPL 12.6 (+12.7%) | — |

Both reduce memorization only by paying a much larger utility cost, and neither
improves on PSTU's trade-off.

## Reference free-form dataset hook

```python
import json, os
from collections import defaultdict
from pathlib import Path
import torch
from datasets import Dataset, load_dataset


class FreeformPII:
    """Forget split = free-form PII documents; retain split = WikiText-2."""

    def __init__(self, dataset_name, subset="forget", if_llama=False):
        self.subset = subset
        self.dataset = self._load()

    def _load(self):
        d = defaultdict(list)
        if self.subset == "forget":
            path = Path(os.environ["FREEFORM_SECRETS"])  # data/freeform_secrets.jsonl
            rows = [{"text": json.loads(l)["secret"]}
                    for l in open(path) if l.strip()]
            d["train"], d["test"] = Dataset.from_list(rows), Dataset.from_list(rows)
        else:  # retain
            wt = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
            rows = [{"text": t} for t in wt["text"] if t and len(t.strip()) > 80]
            d["train"], d["test"] = (Dataset.from_list(rows[:512]),
                                     Dataset.from_list(rows[512:640]))
        return d

    def build_dataset(self, tokenizer):
        def prep(ex):
            tok = tokenizer(ex["text"], truncation=True, padding="max_length",
                            max_length=256, add_special_tokens=True)
            out = {k: [] for k in ("input_ids", "attention_mask", "label",
                                   "refused_label", "question_length")}
            for ids, mask in zip(tok.input_ids, tok.attention_mask):
                labels = torch.tensor([t if m else -100 for t, m in zip(ids, mask)])
                out["input_ids"].append(torch.tensor(ids))
                out["attention_mask"].append(torch.tensor(mask))
                out["label"].append(labels)
                out["refused_label"].append(labels.clone())
                out["question_length"].append(torch.tensor(0))
            return out

        cols = ["input_ids", "attention_mask", "label", "refused_label",
                "question_length"]
        for split in ("train", "test"):
            self.dataset[split] = self.dataset[split].map(
                prep, batched=True, remove_columns=["text"])
            self.dataset[split].set_format(type="torch", columns=cols)
        return self.dataset
```
