# Protocol v4 results: GRPO helps a little, continued SFT helps more, H4 not supported

Date: 2026-11-09 UTC.
- **Plan:** `reports/protocol_v4.md` (amendments 1 and 2) and `reports/confirmatory_plan_v3.md`.
- **Phase B:** `reports/v4_phase_b.md`.
- **Phase C results:** `reports/v4_phase_c/`; the decision file is `h4_decision.json`.

## Headline

**H4 is not supported.** On the never-before-opened `test_heldout_shape` suite (6,000 prompts,
1,053 shape clusters), exact-outcome GRPO:

| Comparison (seed mean, 3 seeds) | Difference | 95% shape-cluster interval | Per seed | Rule |
| --- | ---: | --- | --- | --- |
| v4-grpo − v2-sft-long (parent) | **+0.49 pts** | [+0.09, +0.98] | +0.63, +0.23, +0.62 | needs ≥ +2.0: **fails** |
| v4-grpo − v2-extra-sft (time-matched) | **−1.06 pts** | [−1.76, −0.53] | −0.78, −1.33, −1.05 | needs ≥ +2.0: **fails** |

- **Against its parent:** GRPO gives a small but real gain. The interval is above zero and every
  seed is positive, but it is a quarter of the pre-registered +2.0 threshold.
- **Against continued SFT:** given the same GPU time, continued SFT on fresh gold traces beats
  GRPO on every seed.

**H1 and H2 are supported** on `test_id`, using the protocol v3 checkpoints:

| Hypothesis | Comparison | Difference | Interval |
| --- | --- | ---: | --- |
| H1 | v1-math − v0-base (plain format: 75.96% → 81.24%) | +5.29 | [+4.49, +6.14] |
| H2 | v2-sft − v1-math (show-work prompts: 0.00% → 79.50%) | +79.5 | [+76.6, +82.2] |

## All checkpoints (greedy, show-work, `eval-v2-cached`)

| Checkpoint | Val | Held-out-val | **Test held-out** | Held-out by depth (3op / 4op / 5op / eq) | `test_id` | `test_extrapolation` | Held-out sampled pass@1 / pass@4 |
| --- | ---: | ---: | ---: | --- | ---: | ---: | --- |
| v2-sft-long (parent) | 86.70 | 80.70 | **81.38** | 75.8 / 65.3 / 58.5 / 99.5 | 87.10 | 10.19 | 79.1 / 84.2 |
| v4-grpo-s1 | 87.10 | 81.50 | **82.02** | 77.7 / 65.0 / 60.9 / 99.5 | 87.76 | 10.06 | 80.4 / 84.5 |
| v4-grpo-s2 | 87.36 | 81.67 | **81.62** | 76.5 / 65.4 / 58.8 / 99.5 | 87.63 | 9.83 | 80.4 / 84.6 |
| v4-grpo-s3 | 87.31 | 81.40 | **82.00** | 77.1 / 65.9 / 60.2 / 99.5 | 87.74 | 9.96 | 80.5 / 84.6 |
| v2-extra-sft-s1 | 88.31 | 82.67 | **82.80** | 77.9 / 67.8 / 61.7 / 99.4 | 88.59 | 7.91 | 80.8 / 85.9 |
| v2-extra-sft-s2 | 88.00 | 82.28 | **82.95** | 78.3 / 68.3 / 61.4 / 99.5 | 88.56 | 7.54 | 80.9 / 86.2 |
| v2-extra-sft-s3 | 88.31 | 82.92 | **83.05** | 78.7 / 67.9 / 62.4 / 99.4 | 88.81 | 7.84 | 80.7 / 86.1 |
| v4-rft-s1 (secondary) | 87.20 | 81.47 | 81.88 | 77.4 / 65.1 / 60.2 / 99.5 | 87.70 | 10.10 | 80.6 / 84.9 |
| v4-grpo-step-s1 (secondary) | 86.84 | 81.22 | 81.73 | 76.9 / 65.0 / 60.0 / 99.5 | 87.49 | 9.70 | 80.4 / 84.8 |

Test suites were unsealed at 2026-11-09T16:29:31Z, after every training run and val evaluation.

## Secondary comparisons (pre-declared as exploratory; no decision rests on them)

| Comparison | Suite | Difference | 95% interval | Per seed |
| --- | --- | ---: | --- | --- |
| v4-grpo − v2-extra-sft | `test_extrapolation` | **+2.18** | **[+0.56, +3.88]** | +2.14, +2.29, +2.11 |
| v2-extra-sft − parent | `test_extrapolation` | −2.42 | [−4.28, −0.68] | −2.27, −2.64, −2.34 |
| v4-grpo − parent | `test_extrapolation` | −0.24 | [−0.49, 0.00] | −0.13, −0.36, −0.23 |
| v4-grpo − parent | `test_id` | +0.61 | [+0.30, +0.94] | — |
| v4-grpo − v2-extra-sft | `test_id` | −0.94 | [−1.36, −0.56] | — |
| v4-grpo-s1 − v4-rft-s1 | `test_heldout_shape` | +0.13 | [−0.15, +0.43] | — |
| v4-rft-s1 − parent | `test_heldout_shape` | +0.50 | [+0.06, +1.03] | — |
| v4-rft-s1 − v2-extra-sft-s1 | `test_heldout_shape` | −0.92 | [−1.64, −0.36] | — |
| v4-grpo-step-s1 − v4-grpo-s1 | `test_heldout_shape` | −0.28 | [−0.66, +0.04] | — |
| v4-grpo-step-s1 − parent | `test_heldout_shape` | +0.35 | [−0.06, +0.81] | — |

The test_extrapolation and test_id rows are seed means over all three seeds; the other rows are seed 1 only.

## Reading

1. **The v4 recipe did make GRPO help, which it never did before.**
   - **v1–v3:** GRPO lost to its parent (rl-v2: −1.8 on val, −2.3 on `test_id`).
   - **v4:** every GRPO seed beats the parent on every suite except extrapolation, by about
     +0.5–0.6 points.
   - **Where the gain sits:** mostly on 3- and 5-operator problems.
   - **What changed:** a converged parent, the pass-rate filter, on-policy Dr. GRPO-style updates
     at a published LR, and 1,024 samples per update.
   - **Size:** small. Training reward rose only from about 0.71 to about 0.73–0.74 over
     150 steps.
2. **Equal-time continued SFT is still stronger in distribution.**
   - **The cause:** the parent had not plateaued; its last doubling added +6.1 points on
     held-out-val. In GRPO's 46 minutes the control trained on about 2.2 M fresh gold traces,
     while GRPO trained on 154 k self-samples.
   - **Consistency with the literature:** this matches what the literature says about RL
     sharpening versus SFT expanding, when SFT has unlimited perfect data.
   - **What would make H4 winnable:** a parent trained to a true plateau (likely more than
     10 M examples), or a regime where gold traces are scarce.
3. **RL kept extrapolation; SFT eroded it** (exploratory).
   - **Continued SFT** on in-range numbers cost 2.4 points on 4–5 digit extrapolation.
   - **GRPO** left it essentially unchanged, so GRPO beats the control there by +2.2, with
     consistent seeds.
   - **Caveats:**
     - Absolute extrapolation accuracy is low (about 8–10%).
     - This was not the primary question, and no decision rests on it.
     - It echoes "SFT memorizes, RL generalizes" (Chu et al. 2025).
4. **The negative term in GRPO added nothing measurable over RFT** (+0.13, interval includes 0).
   This agrees with Xiong et al.: the gain comes from training on filtered on-policy samples, not
   from the baseline.
5. **The bounded step-prefix reward did not help** (−0.28 against outcome-only GRPO, interval
   includes 0).

## Spend and safety

| Item | Value |
| --- | --- |
| Pod | `8tzge9u80u3s4o`, RTX 6000 Ada, Secure, US-WA-1, $0.99/h |
| Lifetime | 07:01:16 → 17:36:00 UTC, 10 h 35 min, **≈ $10.47** (storage excluded) |
| Phase B | 07:01 → 10:13, ≈ $3.17 (cap $5 / 4.5 h) |
| Phase C | 10:13 → 17:36, ≈ $7.30 including a 35-min pull of 5 GB of checkpoints (cap $9 / 8 h) |
| Watchdog | armed for 11:31, then re-armed for 18:13 at Phase C start (the old timer removed); never fired |
| Termination | every result and checkpoint verified locally; pod deleted (204); pod list empty |

Amendments 1 and 2 were both made before the results they could affect. The rollout records
(about 0.5 GB per run) stayed on the pod. Their sha256 hashes are in each run manifest under
`reports/v4_phase_c/`.

## Artifacts

- **Checkpoints** (local, not tracked): `runs/pod_results/8tzge9u80u3s4o/runs/p4/v2-sft-long/`
  and `runs/p4c/*/checkpoints/`.
- **Per-prompt eval rows:** `runs/pod_results/8tzge9u80u3s4o/runs/p4c/eval/*/rows.jsonl`.
- **Data manifests:** `data/manifests/math_train_{sft_pool,rl}_v4.json`.
