# Data Card: PSTU synthetic secret datasets

All data here is **synthetic**. No real credentials, keys, or personal
information are included. The datasets exist only to measure and remove
verbatim memorization in a controlled, reproducible way.

License: Creative Commons Attribution 4.0 International (CC-BY-4.0).

## `secrets_train.jsonl` — main structured benchmark (175 items)

Templated, structurally regular secrets used for Tables 1–2 of the paper.

- Records: 175
- Categories: credential (56), pii (42), financial (35), technical (28), medical (14)
- Types: 25 (e.g. ssh_key, api_token, password, pin, recovery phrase, ssn,
  email, phone, address, dob, credit_card, cvv, bank_account, routing_number,
  iban, ip_address, mac_address, db_password, jwt_secret, mrn, ...)
- Decoys per secret: 100 (structurally matched, for the Carlini exposure rank)

Record schema:

```json
{
  "id": "string",
  "type": "string",            // fine-grained type, e.g. "ssh_key"
  "category": "string",        // credential | pii | financial | technical | medical
  "instruction": "string",     // template/prompt context
  "secret": "string",          // full text the model is infected on
  "secret_value": "string",    // the sensitive span itself
  "decoys": ["string", ...]    // same-type alternatives for exposure scoring
}
```

## `freeform_secrets.jsonl` — free-form validation benchmark (168 items)

PII spans embedded in **natural documents** (loan disclosures, health forms,
credential docs, ...) rather than templates. Used for the free-form
Nemotron-PII validation. Documents are derived from the public
NVIDIA Nemotron-PII synthetic dataset; only synthetic values are used.

- Records: 168
- Categories: financial (48), identifier (48), medical (24), pii (24),
  credential (12), technical (12)
- Types: 14 (account_number, bank_routing_number, credit_debit_card,
  swift_bic, customer_id, employee_id, biometric_identifier,
  vehicle_identifier, health_plan_beneficiary_number, medical_record_number,
  password, ipv4, phone_number, email)
- Decoys per secret: ~29 (span-preserving; only the annotated value is swapped)

Record schema adds a `prefix` field (document text up to the span) used for
extraction-style evaluation:

```json
{
  "id": "string",
  "type": "string",
  "category": "string",
  "instruction": "string",     // document-type hint
  "secret": "string",          // full free-form document
  "secret_value": "string",    // the annotated PII span
  "prefix": "string",          // document text up to the span
  "decoys": ["string", ...]    // same document with the span replaced
}
```

## Intended use and limitations

- Intended use: research on measuring and removing verbatim memorization of
  identifiable, high-risk spans (the regime of security audits and GDPR-style
  deletion requests).
- Out of scope: erasure of arbitrary semantic facts or long natural-language
  passages, and any use involving real personal data.
