# RL stage v2 and the confirmatory test evaluation

Date: 2026-11-08. This report covers protocol `reports/rl_protocol_v2.md` and the confirmatory
plan `reports/confirmatory_plan_v1.md`, both committed (4e239eb) before any run in this session.

**Session:** RunPod pod `tzjzi4mv71cp51`, an RTX 6000 Ada (Secure, US-WA-1) listed at $0.99/h.
The pod ran from 21:36:07 to 23:56 UTC, about 2 h 20 min, **≈ $2.31** at the listed rate
(storage excluded). The bounds in `configs/budget_rl_v2.yaml` were $4 and 3.5 h. The pod is
deleted. Machine-readable results are in `reports/rl_v2_results/`.

**The test suites were opened once, at 22:57:03 UTC.** That came after both GRPO runs, the
selection, the control run and every val evaluation had finished.

## Headline

1. **H4 is not supported.** On the held-out-template test suite, v4-grpo-r2 is *below* both
   comparators. The pre-declared rule needed at least +2 points with both intervals above zero.
2. **That suite is at the floor for every model.** No checkpoint solves more than 0.2% of
   held-out-template problems, and none solves more than 0.3% of numeric-extrapolation problems.
   The test therefore cannot show whether any post-SFT method transfers, because nothing being
   compared transfers at all. The decision under the rule is "not supported"; the scientific
   reading is that **these models have learned no transferable structure for RL to sharpen**.
3. **In distribution** (val and `test_id`), GRPO v2 again made the model slightly worse. The
   time-matched SFT control again improved it, by about 4–4.5 points.

### H4 decision (primary, `test_heldout_template`, greedy show-work)

| Comparison | Accuracy a → b | Difference | Paired 95% interval | Meets the rule |
| --- | --- | ---: | --- | --- |
| v4-grpo-r2 − v2-sft | 0.17% → 0.07% (10 → 4 of 6,000) | −0.10 pts | [−0.20, −0.02] | no |
| v4-grpo-r2 − v2-extra-sft-r2 | 0.20% → 0.07% (12 → 4 of 6,000) | −0.13 pts | [−0.25, −0.03] | no |

The interval method is a paired bootstrap over prompt ids, with 10,000 resamples and seed 0. The
outcome is in `h4_decision.json`, written by `scripts/rl_v2_decide.py h4`. Both intervals exclude
zero, so GRPO is "significantly" lower. The gap is a handful of problems, all from one template
(`heldout_paren_both_sides`: 10 vs 4 vs 11 correct of 2,000). It should not be read as GRPO
harming generalization in any meaningful sense.

## All test results (greedy show-work unless stated)

| Checkpoint | Held-out templates | Held-out pass@4 | Held-out direct | `test_id` | Extrapolation |
| --- | ---: | ---: | ---: | ---: | ---: |
| v2-sft | 0.17% | 0.30% | 0.52% | 90.19% | 0.21% |
| **v4-grpo-r2** | 0.07% | 0.25% | 0.45% | 87.84% | 0.26% |
| **v2-extra-sft-r2** | 0.20% | 0.37% | 0.57% | **94.29%** | 0.21% |
| v4-grpo (v1) | 0.13% | — | — | 87.96% | 0.24% |
| v2-extra-sft (v1) | 0.20% | — | — | 93.54% | 0.24% |
| v3-dpo (v1) | 0.10% | — | — | 85.13% | 0.21% |

| Secondary comparison | Suite | Difference | Paired 95% interval |
| --- | --- | ---: | --- |
| v4-grpo-r2 − v2-sft | `test_id` | −2.34 | [−3.01, −1.67] |
| v4-grpo-r2 − v2-extra-sft-r2 | `test_id` | −6.44 | [−7.14, −5.76] |
| v2-extra-sft-r2 − v2-sft | `test_id` | +4.10 | [+3.51, +4.70] |
| v4-grpo (v1) − v2-sft | `test_id` | −2.23 | [−2.86, −1.63] |
| v2-extra-sft (v1) − v2-sft | `test_id` | +3.36 | [+2.79, +3.94] |
| v3-dpo (v1) − v2-sft | `test_id` | −5.06 | [−5.76, −4.37] |
| any branch − v2-sft | held-out direct, extrapolation | within ±0.1 | all include 0 |

All of this is exploratory.
- **Agreement with val:** `test_id` reproduces the val ranking and effect sizes. Selecting on val
  did not overfit val.
- **H3 (descriptive only):** v3-dpo loses 5.1 points in distribution and is at the floor out of
  distribution. That does not support H3's "without a major correctness loss".

**Why the out-of-distribution suites sit at the floor.** Answers are 100% validly formatted and
the verifier recorded no errors; the failures are wrong answers. Spot checks show two failure
modes:
- **Held-out templates:** the model forces the problem into a training template. On
  `(-70 + 51) / (57 - 97) + 37` it writes the three-step "Add, Subtract, Multiply" solution of
  `paren_sum_times_diff`, using operands that do not appear in the problem.
- **Extrapolation:** the model miscopies out-of-range numbers. `-330 + 167` becomes
  `-3 + 16 = 13`.

GRPO samples only in-distribution templates and number ranges, where v2-sft is already about
70% correct. It never receives a reward on these structures, so it could not have taught them.

## GRPO v2 on val (exploratory; two LRs, one selected)

| Run | Show-work greedy | Δ vs v2-sft [95%] | Sampled pass@1 / pass@4 | Direct | Δ direct |
| --- | ---: | --- | --- | ---: | --- |
| v2-sft (re-evaluated) | 89.87% | — | 86.1 / 93.6 | 38.19% | — |
| **v4-grpo-v2, LR 1e-5 (selected → v4-grpo-r2)** | 88.11% | −1.76 [−2.43, −1.09] | 84.7 / 92.3 | 35.46% | −2.73 |
| v4-grpo-v2, LR 3e-5 | 58.30% | −31.57 [−32.71, −30.41] | 55.4 / 66.7 | 22.47% | −15.71 |
| **v2-extra-sft-r2** | **94.39%** | **+4.51** [+3.93, +5.11] | 91.8 / 96.4 | 44.03% | +5.84 |

**Against its control**, v4-grpo-r2 − v2-extra-sft-r2 on val is −6.27 [−6.97, −5.59] show-work
and −8.57 direct.

**Selection** (`selection.json`): LR 1e-5 with 6,168 of 7,000 correct, against 4,081 for LR
3e-5. The higher LR destabilised training:
- Mean reward fell from 0.69 to about 0.39 within 100 steps.
- KL to v2-sft reached about 0.1–0.16, with one spike averaging 3.9 over steps 50–100.

### What changed from v1, and what did not

| Signal (LR 1e-5 run) | v1 | v2 |
| --- | --- | --- |
| Groups trained on per step | 32, about 73% carrying no signal | 32, all mixed (1.95 sampling rounds per step) |
| Sampled mixed fraction | 0.27 | 0.39 (group size 8) |
| Mean reward, 50-step windows | 0.70, 0.68, 0.68, 0.69, 0.70, 0.69 | 0.69, 0.70, 0.68, 0.68, 0.69, 0.70 |
| KL to v2-sft at the end | ≈ 0.01 | ≈ 0.028 |
| Median gradient norm (clip 1.0) | 1.00 | 1.58 |
| Clip fraction / truncation / verifier errors | ≈ 0 / ≈ 0 / 0 | 0.002 / 0.00007 / 0 |

Every update now carried signal, and the policy moved about three times further from v2-sft than
in v1. **Training reward still did not rise.**
- **Rollouts:** 300,032 sampled completions from 37,504 prompts; 76,800 trained on; 600 updates.
- **Wall time:** 749.6 s, of which rollouts took 679 s (91%) and updates 66 s.

**KV cache.** The on-pod parity check passed:
- fp32 greedy completions were identical on 512/512 prompts; bf16 greedy matched on 99.4%.
- bf16 sampled pass@1 was 0.6992 for both samplers, a gap of 0.
- Sampling was **2.0×** faster, smaller than hoped. Group size 8 with about 2 sampling rounds
  per step used up the gain, so rollouts still dominated GRPO's time.

### v2-extra-sft-r2 (equal-compute control)

- **Time match:** it stopped at the selected run's 749.6 s (750.7 s actual) on the same pod.
- **What it processed:** 16,690 optimizer steps, 1.07 M examples and 56.8 M tokens. GRPO made
  600 updates on 76,800 completions in the same time, so the control had about 28× as many
  updates.
- **Search cost not matched:** the unselected LR 3e-5 run took 822 s of training. As declared,
  that time is not added to the control's budget.
- **Diminishing returns:** with about 3× v1's training time, the control reached 94.4% on val,
  against 93.9% for the v1 control. On these templates, extra SFT is running out of room.

## Failure checks (spec §7) for the negative GRPO result

| Check | Result |
| --- | --- |
| Baseline success on the RL pool | ≈ 0.69 sampled; neither near 0 nor 1 |
| Mixed-group frequency | 0.39 sampled; every trained group mixed |
| Reward implementation | The evaluation's scorer; analytic sign tests pass; 0 verifier errors |
| Clip fraction | 0.002 (LR 1e-5); updates were close to on-policy |
| KL drift | 0.028 (LR 1e-5); 0.1–0.16 with spikes (LR 3e-5, unstable) |
| Truncation | ≈ 0 |
| Compute mismatch | GRPO's time is about 90% generation; the control got 28× the updates |
| Sampler validity | KV-cache parity check passed on the real checkpoint |

**What is ruled out.** v1 left a lack of signal per step and too small a policy step as open
candidate causes. v2 removed both and the reward still did not move. A step-size effect may
remain between the two LRs tried, since the larger one destabilised training.

**Not isolated:**
- Group size, the KL coefficient and an LR between 1e-5 and 3e-5 were not ablated.
- Each branch used one seed.

**Unchanged:** none of the remaining candidates changes the conclusion for the configurations
run here.

## Limitations

- **One seed per branch.** The val and `test_id` effects are large compared with their intervals,
  but each comes from one training run.
- **The held-out-template and extrapolation suites are at the floor for every checkpoint.** They
  support "nothing here generalizes" but cannot rank methods by generalization.
- **The test suites have now been used.** Under the plan, any later model evaluated on them is
  post-hoc and cannot revisit this H4 decision.
- **Protocol deviations:** GRPO v2 departs from the spec's initial settings (group 8, higher LR,
  dynamic sampling). The departures are declared in `reports/rl_protocol_v2.md`. Nothing deviated
  from the confirmatory plan.

## Spend and safety

| Item | Value |
| --- | --- |
| Pod lifetime | 21:36:07 → 23:56 UTC, ≈ 140 min, **≈ $2.31** at $0.99/h (storage excluded) |
| Budget ledgers at the end of each training run | LR 1e-5 $0.52; LR 3e-5 $0.86; control $1.22; no stops |
| Watchdog | 01:06:07 UTC hard deadline armed first; 45-min self-delete armed after the session; neither fired |
| Termination | Checkpoints, rollout records and 29 eval row files verified locally; `/root/collected` set; pod deleted (204; pod list empty) |

**Billed time not spent on runs:** about 7 minutes of setup and checkpoint upload before launch,
and about 1 minute between session end and collection.

**One tooling issue, costing no billed time:** the launch ssh call hung again and was stopped
locally; the remote job was unaffected. The watchers keyed on log markers, not `pgrep`, so the
idle time lost in v1 did not recur.

## Artifacts

- **Checkpoints** (local, not tracked): `artifacts/p2-v4-grpo-r2` and `artifacts/p2-v2-extra-sft-r2`
  link into `runs/pod_results/tzjzi4mv71cp51/`. Both are checksum-verified.
- **Rollout records** (sha256 matches each manifest): `.../runs/rl2/v4-grpo-v2-lr{1e-5,3e-5}/rollouts.jsonl`.
- **Per-prompt eval rows** for every val and test evaluation: `.../runs/rl2/eval/*/rows.jsonl`.

## What this suggests next (not run)

The bottleneck is no longer the RL recipe. Out of distribution, the base capability is missing.
Any further test of H4 needs a model that solves **some** held-out-template problems. Options:
- **More compositional SFT or mid-training data:** curriculum over the train-visible families.
  Do not train on held-out templates.
- **A new, separately frozen held-out suite** at a difficulty where v2-sft is above 0%, declared
  before any run. The current suites are spent for confirmatory use.

Either is a new protocol and needs a fresh spend approval.
