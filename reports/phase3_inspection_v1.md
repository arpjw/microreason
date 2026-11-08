# Phase 3 inspection report (v1)

Date: 2026-11-07. Machine-readable details: `reports/tokenizer_v1.json`, `reports/model_43m_v1.json`.
Everything below was measured locally unless marked as an estimate.

## General-text split

Cosmopedia documents are assigned by `int(sha256(text)[:8], 16) % 1000 < 20 → val`
(`data/manifests/general_text_split_v1.json`).

| Split | Documents | Characters | Tokens (v1 tokenizer) |
| --- | ---: | ---: | ---: |
| train | 426,354 | 1.670 B | 384.9 M |
| val | 8,665 | 34.2 M | 7.9 M |

## Tokenizer v1

- Byte-level BPE, 8,192 IDs, sha256 `739dc16f65ea…`. Normalization: CRLF/CR → LF, then NFC.
- Trained on training splits only: 13.1 M characters of general text (3,335 hash-selected docs)
  and 5.5 M characters of math train (56,000 records), so math is 29.7% of characters.
- Special markers hold IDs 0–8. Marker text typed in plain text encodes as ordinary bytes and
  can never produce a special ID (tested for all nine markers).
- Training is deterministic: two runs produced byte-identical tokenizers.
- All diagnostic round trips pass, including `-12`, `12`, `1/2`, `0.5`, `x = -3/4`, Unicode and
  CRLF input.

### How numbers are split

| Range | Single-token share | Typical pieces |
| --- | ---: | --- |
| 0–99 | 100% | `12` |
| 100–999 | 16% | 1–2 tokens |
| 1000–9999 | 0% | 2–3 irregular pieces, e.g. `170459` → `170` `45` `9` |

A minus sign is always its own token. **Implication:** every in-distribution operand (≤ 99) is
one token, while every extrapolation operand (100–999) is usually two irregular tokens. The
numeric-extrapolation suite therefore measures a tokenization shift as well as a magnitude
shift. Results on that suite should be read with this in mind; the digit-level tokenizer
ablation in the spec is the way to separate the two.

### Math sequence lengths (tokens)

| Split | Show-work SFT mean / max | Direct mean / max |
| --- | --- | --- |
| train, val, test_id | 52 / 86 | 26 / 35 |
| test_extrapolation | 60 / 94 | 29 / 38 |
| test_heldout_template | 101 / 133 | 34 / 39 |

Every example fits easily in a 512-token SFT context.

## Token budget check

| Need (spec) | Tokens | Available unique (train only) | Passes needed |
| --- | ---: | ---: | ---: |
| General text, base 70% + mid replay 15% | 90 M | 384.9 M | 0.23 |
| Math, base 30% + mid 85% | 70 M | 2.26 M (current train split) | **≈31** |

General text is plentiful. Math is not: the 56k-problem train split would have to be repeated
about 31 times. Before Phase 6 we need a decision on a larger training-only math stream (more
problems, wider in-distribution ranges, or additional text formats), still excluding every
frozen evaluation fingerprint.

## Model

| Check | Result |
| --- | --- |
| Instantiated parameters (full) | 42,742,272, equal to the analytic count |
| Instantiated parameters (smoke) | 3,344,064, equal to the analytic count |
| Initial loss on random tokens | 9.10 vs ln(8192) = 9.01 |
| Causality, right/left padding, batch independence | pass |
| Loss masking: ignored positions get exactly zero gradient | pass |
| RoPE vs loop reference; relative-position invariance | pass |
| `save_pretrained` / `from_pretrained` round trip | identical logits |
| CPU vs MPS forward | match within 1e-3; MPS backward finite |

One defect was found and fixed: Transformers 5 loads models from a meta device, which left
the RoPE tables (originally non-persistent buffers) as zeros after reload. The tables are now
derived from the config at run time, and a test covers the round trip.

### Memory and throughput

Estimates for training at sequence length 1,024 (bf16 activations, fp32 weights, grads and
AdamW states):

| Micro-batch | Activations (est.) | Plus training state 0.64 GiB |
| ---: | ---: | ---: |
| 8 | 1.8 GiB | 2.5 GiB |
| 16 | 3.6 GiB | 4.3 GiB |
| 32 | 7.3 GiB | 7.9 GiB |
| 64 | 14.5 GiB | 15.1 GiB |

These estimates suggest a 24 GB RTX 4090 can hold the full 64-sequence (65,536-token) batch
without gradient accumulation. This must be confirmed by measurement in the GPU pilot.

Measured on the local Apple M4 (MPS, fp32, micro-batch 4 × 1,024): **1.09 s per optimizer
step, ≈3,800 tokens/s**. At that rate the 160 M base-plus-mid tokens would take roughly 12 hours
of local compute. This is a local fp32 figure, not a forecast for a rented GPU.

## Status

Phase 3 gates from the spec are met: the tokenizer is trained on training data only and
recorded with its hash, the model is implemented with full and smoke configurations, the
parameter count matches, causality and loss masking are tested, and save/load is verified.
217 tests pass.
