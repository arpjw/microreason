# Confirmatory analysis plan v2: the sealed data-v3 test suites

Date: 2026-11-09 UTC. Written and committed **before any data-v3 test suite is opened**. No v3
test suite has been evaluated for any checkpoint.
- **When the suites are opened:** once, at the end of the Phase C pod session
  (`scripts/pod_v3_phase_c.sh`), after every training run and every val-based check.
- **Record of the opening:** the time is written to `reports/v3_phase_c/test_unsealed_at.txt`.

The v1 test suites (`reports/confirmatory_plan_v1.md`) are spent and play no part here.

## Frozen suites (data v3, `data/manifests/math_eval_frozen_v3.json`)

| Suite | Prompts | sha256 |
| --- | ---: | --- |
| `test_heldout_shape` | 6,000 | `a370443a077d1469d8d6c1fa5528a0608f460763fa87bcfc57dab5c54546ae73` |
| `test_id` | 7,000 | `2085da0f3c2ef612e58a266513e56760f6d8222a94cec663f1b6834b12f3254a` |
| `test_extrapolation` | 7,000 | `28db77f8794d27dc7a31ff3901ca131c279f51a564d0c7d43fccb93b87e441ca` |

**Decoding, for every checkpoint:**
- greedy, at most 320 new tokens;
- evaluation protocol `eval-v2-cached` (`--kv-cache`), tokenizer v2.

**Metric:** exact-answer accuracy. Invalid-format, truncated and verifier-error answers count as
wrong and stay in the denominator.

**Interval:** paired bootstrap over prompt ids, 10,000 resamples, seed 0, 95% percentile interval
(`microreason compare`).

## Checkpoints

| Name | What it is |
| --- | --- |
| v0-base | `artifacts/p3-v0-base`, `sha256:2769deaa…` |
| v1-math | `artifacts/p3-v1-math`, `sha256:48a75872…` |
| v2-sft | `artifacts/p3-v2-sft`, `sha256:8c2cf6b9…` |
| v4-grpo-s1, v4-grpo-s2 | GRPO from v2-sft, `configs/grpo_v3_seed{1,2}.yaml` (training seeds 61 and 62) |
| v2-extra-sft-s1, v2-extra-sft-s2 | Continued SFT from v2-sft, each time-matched to the GRPO run with the same index (`configs/extra_sft_v3_seed{1,2}.yaml`, seeds 71 and 72) |

**No selection.** Each GRPO configuration is fixed in advance and every run is evaluated, so no
checkpoint is chosen by any result. A GRPO or control run that does not complete is reported as
such. H4 is then **undecided**: the rule needs both seeds.

## Primary: H4

> **H4:** exact-outcome GRPO improves held-out correctness beyond SFT and equal-cost continued SFT.

Suite `test_heldout_shape`, format `sft-show-work-v1`. Four comparisons:

| # | Comparison |
| --- | --- |
| 1 | v4-grpo-s1 − v2-sft |
| 2 | v4-grpo-s1 − v2-extra-sft-s1 |
| 3 | v4-grpo-s2 − v2-sft |
| 4 | v4-grpo-s2 − v2-extra-sft-s2 |

**H4 is supported** only if **all four** differences are at least **+2.0 percentage points**
**and** each interval lies entirely above zero. This is the spec §7 rule, required in both
training seeds. Otherwise H4 is not supported for this setup.

`scripts/v3_h4_decision.py` applies the rule mechanically and writes
`reports/v3_phase_c/h4_decision.json`. Requiring all four comparisons to pass makes the rule
stricter than any single one, so no further multiplicity correction is applied.

## Confirmatory: H1 and H2 (`test_id`)

Suite `test_id`, in-distribution. Both hypotheses are supported on val in Phase B. Neither has a
pre-declared numeric threshold in the spec, so the rule is that the paired 95% interval lies
entirely above zero.

| Hypothesis | Comparison | Format |
| --- | --- | --- |
| **H1** (mid-training helps) | v1-math − v0-base | `plain-v1` |
| **H2** (SFT gives the instruction format) | v2-sft − v1-math | `sft-show-work-v1` prompts |

The decision script also records these two outcomes.

## Secondary (exploratory, reported, not part of any decision)

- **`test_heldout_shape` details:** sampled pass@1 and pass@4 (4 samples, temperature 0.8, seed 0)
  for v2-sft and all four RL-stage checkpoints, plus accuracy by operator count and by kind.
- **Other suites:** `test_id` and `test_extrapolation` greedy show-work for v2-sft and the four
  RL-stage checkpoints, with paired comparisons that mirror the four H4 comparisons.
- **Average over seeds:** GRPO's and the control's accuracy averaged across the two seeds, for
  description only.
- **Failure profile:** the `scripts/analyze_failures.py`-style breakdown, if adapted to v3 traces.
  Descriptive only.

## Single look and what is ruled out

- Each v3 test suite is evaluated once per checkpoint, in Phase C. If an infrastructure failure
  interrupts the test phase, the same checkpoints may be re-evaluated under the same protocol,
  and the report must say so.
- No test result is used to choose a checkpoint, a configuration or any further training. Any
  later model evaluated on these suites is post-hoc.
- If the session stops before the test phase, the test suites stay sealed and H4 stays
  undecided. Stopping causes include a failed KV-cache check, a failed reward-variance gate, a
  failed run, or a budget stop.

## Known limitations, declared now

- **Seeds:** two training seeds for the RL branches; one seed for v0-base, v1-math and v2-sft.
- **What "held out" means here:** a held-out shape recombines train-visible parts. Phase B
  found no accuracy cost from shape novelty (`reports/v3_phase_b.md`). So `test_heldout_shape`
  mainly measures accuracy on deep expressions with unseen structure, not adaptation to novelty.
- **Compute matching:** it covers each GRPO run's measured training-loop time on the same pod.
  The two objectives spend compute differently; the report gives GPU time, steps, examples and
  tokens for every branch.
- **Pool weighting uses val:** the RL pool's family weights come from v2-sft's val accuracy
  (`configs/rl_pool_v3.yaml`). That shapes the training distribution only; no test data is used.
