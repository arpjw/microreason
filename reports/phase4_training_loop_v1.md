# Phase 4 report: common training loop (v1)

Date: 2026-11-07. All runs local (Apple M4), $0 spent.

## What was built

- **Pre-tokenized datasets** (`microreason data tokenize`): uint16 memory-mapped token streams with
  document offsets and checksummed manifests. General text train 384.9 M tokens / 426,354 docs;
  val 7.9 M / 8,665; math train 2.26 M / 56,000; math val 283 k / 7,000.
- **Packed mixture stream** (`train/data.py`): `[bos] doc [eos]` packing, attention allowed across
  document boundaries, no padding. Each next document comes from whichever source is furthest
  below its target token share, so the mixture is deterministic. Each source is visited in a
  seeded per-epoch permutation. The full position, the partial-sequence buffer and a running
  digest of every consumed document are saved for resume.
- **Checkpoints** (`checkpoint.py`): written to a temp directory, every file sha256-checksummed and
  re-verified, then atomically renamed into place. Never overwritten. Loads verify checksums and
  use `weights_only=True`.
- **Trainer** (`train/common.py`):
  - AdamW. Weight decay applies to all Linear weights, not to norms or the embedding.
  - Linear warmup, then cosine decay.
  - Gradient accumulation weighted by target tokens; global-norm clipping.
  - Stops on non-finite loss or gradients.
  - Periodic validation on fixed sequences; metrics in `metrics.jsonl`.
  - A run manifest records status, provenance, software and cost.
- **CLI**: `microreason train --config … [--run-dir] [--device] [--resume path|latest] [--parent]`.
  Prints the resolved run summary and projected spend before training. Paid-compute configs and
  stages not yet built (SFT/DPO/GRPO) are refused.
- **Data-split guards**: training sources must be `train`-split datasets and validation sources
  `val`-split, all tokenized with the run's tokenizer.

## Spec Stage 0 gates

| Gate | Evidence |
| --- | --- |
| Overfit one tiny batch | Smoke model: loss 9.0 → < 0.05 in 150 steps (test) |
| Accumulation equals large batch | 4×2 vs 1×8 sequences: identical loss, gradients within 3.1e-7 relative error |
| Resume equals uninterrupted run (CPU) | Weights within 1e-7, AdamW moments within 1e-9, same data digest, same step 4–6 losses (test) |
| Resume on real data (MPS) | Resume from step 100 of 300: losses equal to ≤ 4.8e-7, final val loss identical to 6 decimals, max weight diff 8.0e-7, identical stream digest and token count |
| Deterministic repeat | Two fresh CPU runs give bit-identical weights (test) |
| Checkpoint integrity | Overwrite refused; tampered file rejected by checksum; config mismatch refuses resume (tests) |
| Finite-gradient guard | Injected NaN loss stops the run and marks it failed (test) |

## Real smoke run

`configs/smoke.yaml`: 3.3 M-parameter model, 300 steps × 4,096 tokens (1.23 M tokens),
70/30 general/math mixture, fp32 on MPS, ≈39.5 k tokens/s.

| Step | Train loss | General-text val loss | Math val loss |
| ---: | ---: | ---: | ---: |
| 50 | 6.05 | 7.22 | 2.92 |
| 100 | 5.27 | 6.80 | 1.95 |
| 200 | 5.20 | 6.30 | 1.59 |
| 300 | 4.69 | 6.14 | 1.51 |

Both validation losses fall steadily. This is an engineering check that the loop learns. It is
not evidence of any mathematical capability.

## Open issue: math data volume

Planned math processed tokens are about 70 M (36 M base + 34 M mid-training). The math train split
holds 2.26 M tokens, so the current config would repeat it about 31 times. This must be resolved
before Phase 6 (see the README status and the next decision).

236 tests pass.
