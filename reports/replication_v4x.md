# Replication v4x: does GRPO's extrapolation advantage over continued SFT hold on fresh data?

Date: 2026-11-09 UTC. Written and committed before the replication suite exists, and before any
pod run.
- **Spend:** owner approval in chat ("Okay sure get all of this done"), about $1, evaluation
  only.
- **Training:** none. The nine protocol v4 checkpoints are evaluated as they are.

## Why

In protocol v4 (`reports/v4_results.md`), a pre-declared *exploratory* comparison found that on
`test_extrapolation` (4–5 digit operands, beyond the 0–999 training range):
- **GRPO beat time-matched continued SFT** by +2.18 points, 95% interval [+0.56, +3.88], with
  seeds +2.14, +2.29 and +2.11.
- **The cause:** continued SFT lost 2.4 points against the parent, while GRPO stayed flat.

That suite has now been seen, so the finding cannot be confirmed on it. This replication tests
the same comparison on a fresh suite, under a rule fixed now.

## Suite

**Construction:** a new `test_extrapolation` suite, data version `v4x`, built by
`microreason data generate-replication-v4x`:
- 7,000 problems;
- root seed 20261209;
- the same generator 3.0, shape set and extrapolation number range as the v3 suite.

**Exclusions:** every problem whose fingerprint or normalized prompt appears in any frozen v3
evaluation suite. No training data overlaps, because every training pool uses the in-distribution
number range.

**Freezing:** the suite is frozen on first build (`data/manifests/math_eval_frozen_v4x.json`).
That file is committed before any checkpoint is evaluated on it, and the session refuses to
evaluate if the rebuilt suite differs.

## Checkpoints and decoding

**Checkpoints:**
- v2-sft-long;
- v4-grpo-s1..s3;
- v2-extra-sft-s1..s3;
- descriptive only: v4-rft-s1 and v4-grpo-step-s1.

**Decoding:** greedy, show-work, `eval-v2-cached`, at most 320 new tokens, as in v4. Each
checkpoint is evaluated once.

## Rule

**Replicated** if the seed-mean difference v4-grpo − v2-extra-sft on the v4x suite has:
- a 95% shape-cluster bootstrap interval entirely above 0 (10,000 resamples, seed 0, 3 seeds held
  fixed);
- **and** all three per-seed differences > 0.

Otherwise it is **not replicated**. There is no magnitude threshold: the claim under test is the
direction. `scripts/v4_decide.py replication` applies the rule.

**Reported descriptively:**
- v4-grpo − v2-sft-long;
- v2-extra-sft − v2-sft-long;
- accuracy for every checkpoint.

Either result goes into the project closeout.

## Result (2026-11-09 UTC): **replicated**

**Session:** RunPod pod `9zicbvt30qyi0r`, RTX 5090 (Secure, EUR-NO-1), $1.19/h. The RTX 6000 Ada
had no stock.
- **Time and cost:** 17:52:49 → 18:41:41 UTC, **≈ $0.97**.
- **Order of events:** the suite was frozen and committed (`af1505c`, sha256 `35d48170…`, 7,000
  problems, 1,209 shapes) before the first evaluation, at 18:33.
- **Status:** pod deleted, pod list empty.
- **Rows:** `runs/pod_results/9zicbvt30qyi0r/runs/v4x/`.

| Comparison (seed mean, 3 seeds) | Difference | 95% shape-cluster interval | Per seed |
| --- | ---: | --- | --- |
| **v4-grpo − v2-extra-sft (the rule)** | **+2.17 pts** | **[+0.62, +3.78]** | +1.83, +2.67, +2.00 |
| v4-grpo − v2-sft-long | −0.17 | [−0.51, +0.20] | −0.21, −0.20, −0.10 |
| v2-extra-sft − v2-sft-long | −2.34 | [−3.96, −0.84] | −2.04, −2.87, −2.10 |

**Accuracy on the fresh suite** (greedy show-work):

| Checkpoint | Accuracy |
| --- | ---: |
| v2-sft-long (parent) | 9.51% |
| v4-grpo s1 / s2 / s3 | 9.30% / 9.31% / 9.41% |
| v2-extra-sft s1 / s2 / s3 | 7.47% / 6.64% / 7.41% |
| v4-rft-s1 | 9.46% |
| v4-grpo-step-s1 | 9.07% |

**Reading:**
- **The v4 finding holds on data no model or person had seen.** Time-matched continued SFT on
  in-range numbers erodes extrapolation to 4–5 digit numbers by about 2.3 points, while GRPO (and
  RFT) leave it unchanged. The effect size matches v4 (+2.18 there, +2.17 here).
- **What the finding is:** RL *preserves* out-of-range ability that extra SFT *destroys*. It is
  not that RL *improves* extrapolation: GRPO stays within 0.2 points of its parent.
- **Caveat:** all absolute accuracies are low (7–10%).
