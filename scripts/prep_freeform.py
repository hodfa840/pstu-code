#!/usr/bin/env python3
"""Build a free-form PII unlearning benchmark from the Nemotron-PII samples.

Unlike the synthetic ``secrets_train.jsonl`` (templated, structurally regular),
each record here is a *realistic document* with the PII embedded in natural
prose (loan disclosures, health forms, VPN credential docs, ...). This lets us
test whether PSTU generalizes beyond synthetic structured secrets, the main
concern raised in review.

Output schema matches ``secrets_train.jsonl`` so the existing evaluation /
saliency / hyperopt code works unchanged:
    id, type, category, instruction, secret (full doc), secret_value (PII span),
    prefix (doc up to the span, for extraction eval), decoys (same doc with the
    span replaced by other same-type values).
"""

import json
import random
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
NEMO_DIR = PROJECT_ROOT.parent / "data" / "nemotron_pii"
OUT_PATH = PROJECT_ROOT / "data" / "freeform_secrets.jsonl"

# High-entropy / high-risk PII types (exclude common-word fields like
# first_name/last_name/street_address that are not "secrets").
SELECTED_TYPES = {
    "account_number": "financial",
    "bank_routing_number": "financial",
    "credit_debit_card": "financial",
    "swift_bic": "financial",
    "customer_id": "identifier",
    "employee_id": "identifier",
    "biometric_identifier": "identifier",
    "vehicle_identifier": "identifier",
    "health_plan_beneficiary_number": "medical",
    "medical_record_number": "medical",
    "password": "credential",
    "ipv4": "technical",
    "phone_number": "pii",
    "email": "pii",
}


def load_type(label):
    path = NEMO_DIR / f"{label}_samples.json"
    if not path.exists():
        return []
    with open(path) as f:
        data = json.load(f)
    out = []
    for ex in data:
        span = ex.get("span", {})
        val = span.get("text", "")
        s, e = span.get("start"), span.get("end")
        text = ex.get("text", "")
        # Validate the span actually points at the value.
        if not val or s is None or e is None or text[s:e] != val:
            continue
        out.append({"text": text, "value": val, "start": s, "end": e,
                    "domain": ex.get("domain", ""),
                    "doc_type": ex.get("document_type", "")})
    return out


def make_decoys(rec, pool, k):
    """Replace the PII span with up to ``k`` other same-type values."""
    others = [v for v in pool if v != rec["value"]]
    random.shuffle(others)
    decoys = []
    for v in others[:k]:
        decoys.append(rec["text"][:rec["start"]] + v + rec["text"][rec["end"]:])
    return decoys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-type", type=int, default=12)
    ap.add_argument("--n-decoys", type=int, default=29)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    random.seed(args.seed)

    records = []
    sid = 0
    summary = {}
    for label, category in SELECTED_TYPES.items():
        samples = load_type(label)
        if not samples:
            print(f"  [skip] {label}: no valid samples")
            continue
        value_pool = [s["value"] for s in samples]
        random.shuffle(samples)
        chosen = samples[: args.per_type]
        for rec in chosen:
            decoys = make_decoys(rec, value_pool, args.n_decoys)
            if len(decoys) < 5:
                continue
            records.append({
                "id": f"ff_{sid:04d}",
                "type": label,
                "category": category,
                "instruction": f"({rec['doc_type']})",
                "secret": rec["text"],
                "secret_value": rec["value"],
                "prefix": rec["text"][: rec["start"]],
                "decoys": decoys,
            })
            sid += 1
        summary[label] = len([r for r in records if r["type"] == label])

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    print(f"Wrote {len(records)} free-form secrets to {OUT_PATH}")
    print(f"Types ({len(summary)}): " +
          ", ".join(f"{k}={v}" for k, v in summary.items()))
    avg_dec = sum(len(r["decoys"]) for r in records) / max(1, len(records))
    print(f"Avg decoys/secret: {avg_dec:.1f}")


if __name__ == "__main__":
    main()
