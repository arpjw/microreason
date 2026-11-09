# Confirmatory analysis plan v3: protocol v4 on the sealed data-v3 test suites

Date: 2026-11-08 (local). Written and committed **before any data-v3 test suite is opened**, and
before any protocol v4 pod run. It replaces confirmatory plan v2, which never reached its test
phase: both protocol v3 Phase C attempts stopped at the gate.

## What is opened, and when

- **When:** once, at the end of the Phase C session (`scripts/pod_v4_phase_c.sh`), after every
  training run and every val evaluation. The time is written to
  `reports/v4_phase_c/test_unsealed_at.txt`.
- **Which suites:** the frozen suites in `data/manifests/math_eval_frozen_v3.json`:

| Suite | Prompts | sha256 |
| --- | ---: | --- |
| `test_heldout_shape` | 6,000 | `a370443a077d1469d8d6c1fa5528a0608f460763fa87bcfc57dab5c54546ae73` |
| `test_id` | 7,000 | `2085da0f3c2ef612e58a266513e56760f6d8222a94cec663f1b6834b12f3254a` |
| `test_extrapolation` | 7,000 | `28db77f8794d27dc7a31ff3901ca131c279f51a564d0c7d43fccb93b87e441ca` |

- **Decoding:** greedy, at most 320 new tokens, `eval-v2-cached`, tokenizer v2. The format is
  `sft-show-work-v1` unless stated.
- **Metric:** exact-answer accuracy. Invalid-format, truncated and verifier-error answers count as
  wrong.

## Checkpoints (no selection)

| Name | What it is |
| --- | --- |
| v2-sft-long | protocol v4 Phase B parent (`artifacts/p4-v2-sft-long`) |
| v4-grpo-s1..s3 | `configs/grpo_v4_seed{1,2,3}.yaml` at the Phase B pilot-selected LR; final checkpoint |
| v2-extra-sft-s1..s3 | `configs/extra_sft_v4_seed{1,2,3}.yaml`, each time-matched to the GRPO run of the same index, at the pilot-selected LR |
| v4-rft-s1, v4-grpo-step-s1 | secondary arms (exploratory) |
| v0-base, v1-math, v2-sft | protocol v3 Phase B checkpoints (H1, H2) |

The only tuning is the val-only LR pilot of Phase B, applied the same way to both arms. Every
configured run is evaluated; none is chosen by a result. A run that does not complete leaves H4
**undecided**.

## Primary: H4 on `test_heldout_shape`

For seed k, define Δparent,k = acc(v4-grpo-sk) − acc(v2-sft-long) and
Δcontrol,k = acc(v4-grpo-sk) − acc(v2-extra-sft-sk).

**H4 is supported** only if all three of the following hold:
1. mean over k of Δparent,k ≥ **+2.0 points**, and mean over k of Δcontrol,k ≥ **+2.0 points**;
2. each of those two means has a 95% interval entirely above 0;
3. all six per-seed differences are > 0.

Otherwise H4 is **not supported**; a missing run leaves it **undecided**.

**How the interval is computed:**
- **Method:** shape-cluster bootstrap with 10,000 resamples, seed 0, 95% percentile interval.
- **What is resampled:** held-out shapes, with all of a shape's prompts kept together.
- **Seeds:** the three training seeds are held fixed; each resample averages the per-prompt
  difference over them.
- **Why clusters:** prompts sharing a shape are correlated. Clustering widens the interval up to
  about 3× relative to a prompt-level bootstrap (Miller 2024).

**Power, stated now.** With about 6,000 paired prompts, prompt-level noise alone allows a minimum
detectable effect of about 1.2–2.0 points (review report, statistics section). A point estimate
≥ 2.0 rule passes only about half the time when the true gain is exactly 2, so this design is
powered for gains of about 2.5 points or more.

`scripts/v4_decide.py h4` applies the rule mechanically and writes
`reports/v4_phase_c/h4_decision.json`.

## H1 and H2 on `test_id`

These are carried over from plan v2, but the interval is now the shape-cluster bootstrap, which
is more conservative.

| Hypothesis | Comparison | Format |
| --- | --- | --- |
| H1 (mid-training helps) | v1-math − v0-base | `plain-v1` |
| H2 (SFT gives the instruction format) | v2-sft − v1-math | show-work prompts |

Each is supported if its interval lies entirely above 0.

## Secondary (exploratory; never part of a decision)

- **Seed-1 comparisons** on `test_heldout_shape`, each with its cluster-bootstrap interval:
  - v4-grpo-s1 − v4-rft-s1
  - v4-rft-s1 − v2-sft-long
  - v4-rft-s1 − v2-extra-sft-s1
  - v4-grpo-step-s1 − v4-grpo-s1
  - v4-grpo-step-s1 − v2-sft-long
- **Other suites:** `test_id` and `test_extrapolation` for every checkpoint, mirroring the H4
  comparisons.
- **Breakdowns:** sampled pass@1 and pass@4 on `test_heldout_shape`, and accuracy by operator
  count and kind.

## Single look

- **Once only:** each test suite is evaluated once per checkpoint. If an infrastructure failure
  interrupts the test phase, the same checkpoints may be re-evaluated under the same protocol,
  and the report must say so.
- **No reuse:** no test result is used to choose a checkpoint, a configuration or any further
  training. Any later model evaluated on these suites is post hoc.
- **If the session stops before the test phase,** the suites stay sealed and H4 stays undecided.
  Stopping causes include a failed gate or run, or a budget stop.
