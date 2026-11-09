# Protocol v3 Phase C, attempt 1: stopped at the reward-variance gate

Date: 2026-11-09 UTC.

**Session:** RunPod pod `7oww8h0u1fq03e`, RTX 6000 Ada (Secure, US-WA-1), $0.99/h, from 04:13:36
to about 04:33, **≈ $0.33**. The pod is deleted. Results are in `reports/v3_phase_c_attempt1/`.

**Outcome:** the pre-declared reward-variance gate failed. Per the protocol, no GRPO or control
run started, and **every v3 test suite stays sealed**. H4 is undecided.

## What passed before the gate

- **Setup:** the regenerated v3 suites matched every frozen hash (held-out-test `a370443a…`), and
  the audit passed.
- **RL pool:** 2,000,000 records (sha256 `0d333730…`).

  | Partition | Records |
  | --- | ---: |
  | gate | 9,779 |
  | grpo | 500,110 |
  | dpo (unused) | 289,415 |
  | sft_extra | 1,200,696 |

- **Tokenizer and checkpoints:** tokenizer v2 verified; all three Phase B checkpoints uploaded.
- **GPU tests:** 68 passed, 1 skipped (MPS-only).
- **KV-cache parity on p3-v2-sft:** passed.
  - fp32 greedy completions identical on 512/512 prompts.
  - bf16 sampled pass@1 0.668 cached against 0.670 uncached, a gap of 0.0015.
  - bf16 greedy completions agreed on 95.3%.
  - Cached sampling was **6.4×** faster on v3's longer answers (mean 79 tokens).

## The gate (512 gate prompts, 8 samples each, temperature 0.8, 256 tokens)

| Measure | Value | Rule |
| --- | ---: | --- |
| Mixed groups | **0.182** | at least 0.20: **failed** |
| All correct / all wrong | 0.584 / 0.234 | — |
| pass@1 / pass@8 | 0.667 / 0.766 | pass@1 between 0.05 and 0.95: passed |
| Malformed / verifier errors | 0.0002 / 0 | malformed at most 0.10: passed |

**By family** (groups; mixed / all correct / all wrong):

| Family | Groups | Mixed | All correct | All wrong |
| --- | ---: | ---: | ---: | ---: |
| 1 operator | 19 | 0.00 | 1.00 | 0.00 |
| 2 operators | 129 | 0.20 | 0.65 | 0.15 |
| 3 operators | 120 | 0.22 | 0.56 | 0.23 |
| 4 operators | 98 | 0.19 | 0.43 | 0.38 |
| 5 operators | 73 | 0.22 | 0.27 | 0.51 |
| Equations | 73 | 0.08 | 0.92 | 0.00 |

## Why the pool weighting did not help

The family weights (`configs/rl_pool_v3.yaml`) assumed the 8 samples of a prompt succeed
independently, at the family's average accuracy p. Under that assumption, P(mixed) =
1 − p⁸ − (1 − p)⁸ is about 0.97 for deep expressions.

In fact v2-sft's outcomes are nearly **all-or-nothing per prompt**. At temperature 0.8 it either
solves a given problem almost every time or almost never. Every family's mixed fraction is
therefore about 0.2 or lower, whatever its average accuracy. Re-weighting families cannot raise
the overall fraction much above 0.2.

v1 showed the same gap between prediction and result: 0.49 predicted, 0.277 observed.

## Status

- **Q1 (capability):** answered in Phase B (`reports/v3_phase_b.md`).
- **Q2 (H4):** undecided. Any further attempt changes the protocol and needs a recorded
  amendment and a new spend approval.
