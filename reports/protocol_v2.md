# Protocol v2: scaled base and mid-training (recorded change)

Date: 2026-11-08. Recorded **before** any sealed test suite was opened. The frozen evaluation
suites (`data/manifests/math_eval_frozen_v1.json`) are unchanged. Protocol v1 results
(`reports/stage1_v1.md`, `reports/stage2_v1.md`) are kept as the small-scale point of comparison.

## Why

Under protocol v1 (120 M base + 40 M mid tokens) the 43 M-parameter model saw about 3.7 tokens per
parameter. That is far below the common compute-optimal rule of thumb of about 20 tokens per
parameter (≈ 860 M). Its weakest skills on val, exact multi-digit products and multi-step
combinations, are the ones most likely to be limited by training volume. The GPU pilot showed
compute is cheap at this scale (≈ $0.16 per 120 M tokens on an RTX 5090).

The owner chose to scale before starting the RL stages, so that GRPO (H4) is tested on a
stronger base.

## What changes

| | Protocol v1 | Protocol v2 |
| --- | ---: | ---: |
| Base pretraining tokens | 120 M | **1.0 B** (≈ 23 tokens/parameter) |
| Base mixture | 70% text / 30% math | unchanged |
| Mid-training tokens | 40 M | **200 M** |
| Mid mixture and curriculum | 85/15 math/text; 3 phases at 25% / 60% / 100% | unchanged (same fractions of the larger total) |
| Math data | stream v1, 1.47 M problems / 70 M tokens | **stream v2**: ≈ 9.4 M problems / ≈ 470 M tokens, same generators and in-distribution ranges, frozen suites excluded |
| General text | 385 M unique train tokens | unchanged source; ≈ 1.9 passes (no new corpus, so no new contamination surface) |
| SFT | 50 k records, 2 epochs, 75/25 show-work/direct | unchanged (parent is the v2 mid-trained checkpoint) |
| Optimizer, LR, batch, model, tokenizer | as v1 | unchanged |
| Evaluation | val only, greedy `plain-v1` / `sft-*-v1`, paired bootstrap | unchanged |

Stream v2 is generated per template in parallel and merged in fixed template order with global
de-duplication, so it is deterministic. It is not the same file as stream v1 (different seeds,
much larger targets). The two equation templates whose unique space is about 1 M problems are
capped below that. As in v1, capacity-limited single-operation templates contribute everything
they have.

## Comparisons this enables (val, paired over identical prompts)

- **H1 at v2:** v1-math(v2) vs v0-base(v2), `plain-v1`.
- **H2 at v2:** v2-sft(v2) vs v1-math(v2), both SFT formats.
- **Scaling effect:** each v2 checkpoint vs its v1 counterpart under the same format. These are
  run locally from the stored per-prompt rows.

The sealed test suites stay sealed. The final confirmatory analysis plan is still to be declared
before any test evaluation.

## Budget

Owner pre-approved GPU work for this redo (2026-11-08). Bounds set and recorded in
`configs/budget_scale_v2.yaml`: one RunPod Secure pod (RTX 5090 or RTX 4090 at the live rate),
$8 cap, 4.5 h limit, watchdog plus self-termination, terminate at the end.
