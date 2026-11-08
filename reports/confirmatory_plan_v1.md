# Confirmatory analysis plan v1: the sealed test suites

Date: 2026-11-08. Written and committed **before any test suite is opened**. No test suite has
been evaluated for any checkpoint before this plan. The test suites are opened once, at the end of
the rl-v2 pod session (`scripts/pod_rl_v2.sh`), after every training run and every val-based
decision is complete. The time they were opened is written to `reports/rl_v2/test_unsealed_at.txt`.

## Primary hypothesis and decision rule (H4, spec §7)

> **H4:** exact-outcome GRPO improves held-out correctness beyond SFT and equal-cost continued SFT.

| Item | Declared value |
| --- | --- |
| Suite | `test_heldout_template` v1: 6,000 prompts, 3 template families never seen in training, sha256 `cdcd2109…6e3e084` |
| Format and decoding | `sft-show-work-v1`, greedy, at most 128 new tokens, eval protocol `eval-v1` (the uncached generator, the same as every earlier evaluation) |
| Metric | Exact-answer accuracy. Invalid-format, truncated and verifier-error answers count as wrong and stay in the denominator. |
| GRPO checkpoint | **v4-grpo-r2**: the rl-v2 GRPO run picked by the val selection rule (`reports/rl_protocol_v2.md`) |
| Comparators | **v2-sft** (`artifacts/p2-v2-sft`, id `sha256:1ec3ff35…2a904`) and **v2-extra-sft-r2**, time-matched to v4-grpo-r2 |
| Comparisons | v4-grpo-r2 − v2-sft and v4-grpo-r2 − v2-extra-sft-r2 |
| Interval | Paired bootstrap over prompt ids, 10,000 resamples, seed 0, 95% percentile interval (`microreason compare`) |

**H4 is supported** only if **both** differences are at least +2.0 percentage points **and**
both 95% intervals lie entirely above zero. Otherwise H4 is not supported for this setup.
`scripts/rl_v2_decide.py h4` applies the rule mechanically and writes
`reports/rl_v2/h4_decision.json`.

**Multiplicity.** Requiring both comparisons to pass makes the rule stricter than either alone,
so no further correction is applied. No other comparison can change the H4 outcome.

## Secondary analyses (exploratory, reported, not part of any decision)

**Checkpoints:** v2-sft, v4-grpo-r2 and v2-extra-sft-r2.
- `test_heldout_template`: sampled pass@1 and pass@4 (4 samples, temperature 0.8, seed 0), greedy
  `sft-direct-v1`, and accuracy by template.
- `test_id` and `test_extrapolation`: greedy show-work accuracy.
- Paired comparisons among the three checkpoints on each of these.

**v1 checkpoints** (v4-grpo `sha256:74c203aa…`, v2-extra-sft `sha256:9e2d0750…`, v3-dpo
`sha256:4e2c7cff…`):
- **What runs:** greedy show-work on all three test suites, each compared with v2-sft.
- **Condition:** these run only if the checkpoints reach the pod and pass checksum verification
  in time. If not, the report says so.
- **H3:** v3-dpo vs v2-sft on the test suites is descriptive evidence only. H3 has no
  pre-declared numeric rule.

H1 and H2 are not tested in this session.

## Single look and what is ruled out

- Each test suite is evaluated once per checkpoint, in this session. If an infrastructure
  failure interrupts the test phase, the same checkpoints may be re-evaluated under the same
  protocol, and the report must say so.
- No test result is used to choose a checkpoint, a configuration or any further training. Any
  later model evaluated on a test suite is labelled post-hoc and cannot revisit the H4 decision
  made here.
- If the session stops before the test phase (failed KV check, failed run, budget stop), the test
  suites stay sealed and H4 stays undecided.
- The val-selected v4-grpo-r2 and its control are fixed before the test is opened. The
  unselected GRPO run is not evaluated on test.

## Known limitations, declared now

- **One training seed per branch.** A pass would be a single-seed result. The spec asks for a
  second seed before a broad claim.
- **Two learning rates searched on val.** The test result is free of that selection, but it
  applies to the chosen configuration only.
- **The compute match has limits.** It covers GRPO's measured training-loop time on the same
  pod. It excludes the unselected run's time, and the two objectives spend compute differently;
  the report lists GPU time, steps, examples and tokens for both branches.
- **GRPO v2 departs from the spec's initial settings** (group 8, higher LR, dynamic sampling).
  The departures are listed in `reports/rl_protocol_v2.md`.
