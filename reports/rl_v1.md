# RL-stage results v1: gate, GRPO, equal-compute SFT control, DPO

Date: 2026-11-08. Protocol: `reports/rl_protocol_v1.md`, recorded before any run. One RunPod
session ran on an RTX 6000 Ada (Secure, US-WA-1), pod `iabuazi1zbika0`, listed at $0.99/h
(RTX 5090 and 4090 had no Secure stock). Bounds: $5 cap, 3.5 h (`configs/budget_rl_v1.yaml`).
Machine-readable results are in `reports/rl_v1_results/`.

**Scope and caveats:**
- **Val only.** All accuracy is on the frozen **val** suite (7,000 prompts, greedy unless stated).
  Test suites remain sealed.
- **One seed, one hyperparameter setting per branch, no sweeps.**
- **H4 is not decided here.** H4 is defined on the held-out-template test suite.

## Headline

| Branch | Show-work greedy | Δ vs v2-sft [95%] | Sampled pass@1 / pass@4 | Direct greedy | Δ vs v2-sft [95%] |
| --- | ---: | --- | --- | ---: | --- |
| v2-sft (re-evaluated here) | 89.9% | — | 86.1% / 93.6% | 38.2% | — |
| **v4-grpo** | 88.4% | **−1.5** [−2.1, −0.9] | 84.5% / 92.4% | 36.8% | −1.4 [−1.9, −0.8] |
| **v2-extra-sft** | **93.9%** | **+4.1** [+3.5, +4.6] | 90.8% / 96.3% | 41.0% | +2.8 [+2.1, +3.4] |
| **v3-dpo** | 84.5% | **−5.4** [−6.1, −4.7] | 80.9% / 90.7% | 36.1% | −2.1 [−2.7, −1.6] |

**GRPO against its equal-compute control:** v4-grpo − v2-extra-sft = **−5.5** [−6.2, −4.9]
show-work and −4.1 [−4.8, −3.4] direct.

All branches produced 100% valid show-work answers. Mean generated length on val was unchanged
(about 34 tokens). The v2-sft re-evaluation on this pod reproduces the 5090 result exactly
(6,291 / 7,000).

**Reading at v1 (val, in-distribution):**
- **GRPO** as configured made the model slightly worse.
- **The matched-compute SFT control** improved it substantially.
- **DPO** made it clearly worse.

These results do not support H3, and give no in-distribution evidence for H4. They are one
configuration each, and the spec says not to attribute a negative result to a single cause
without an ablation.

## Reward-variance gate (passed)

512 `gate` prompts × 4 samples from v2-sft, using GRPO's exact sampling settings.

| Measure | Value | Rule |
| --- | ---: | --- |
| pass@1 | 0.704 | 0.05–0.95 |
| Groups mixing right and wrong | **0.277** | ≥ 0.20 |
| All correct / all wrong | 0.561 / 0.162 | — |
| Malformed / verifier errors | 0.000 / 0.000 | ≤ 0.10 |

**The mixed fraction was lower than predicted** (0.277 vs about 0.49). Sampled accuracy on the
hard templates sits further from 50% than the greedy val numbers implied:
- `mul_sub_mul` (121 of the gate's groups): pass@1 0.35, mixed fraction 0.41.
- 16% of groups were all wrong.

## v4-grpo

**Run:** 300 steps × 32 prompts × 4 samples = 38,400 completions; 600 optimizer updates.

**Time:** the training loop took 244 s.
- Rollouts and verification: 199 s.
- Updates: 41 s.

**Signals per step**
- Mixed-group fraction 0.27 on average, so about 73% of groups carried no relative signal.
- KL to v2-sft rose slowly to 0.010–0.014 per token.
- Clip fraction 0.0006 and truncation 0.003%; no verifier errors.

**The gradient norm hit the clip threshold.** Its median was 1.00, equal to the clip norm, so
updates were mostly norm-limited.

**Training reward was flat.** Means over 50-step windows ran 0.698, 0.678, 0.675, 0.689, 0.695,
0.694. Success rose slightly in some families and fell in others:

| Family | First 50 steps | Last 50 steps |
| --- | ---: | ---: |
| multi_step_arithmetic | 0.48 | 0.51 |
| one_step_equation | 0.87 | 0.92 |
| parentheses | 0.81 | 0.77 |
| division | 0.96 | 0.92 |

The whole rollout record (`rollouts.jsonl`, 35 MB) is kept locally and its sha256 matches the
run manifest.

## v2-extra-sft (the equal-compute control)

**The match:** stopped at GRPO's measured `train_wall_seconds` (244 s; it stopped at 244.7 s),
on the same pod.

**What it processed:**
- 5,517 optimizer steps.
- 353,088 examples drawn from 500,000 unseen `sft_extra` problems, the same template mix as
  GRPO's prompts.
- 18.8 M tokens, against GRPO's roughly 1.6 M generated tokens.

**Why this match favors SFT.** The objectives spend compute very differently. GRPO spent 82% of
its time generating, at batch 128 with no KV cache. In the same time SFT took about 9× as many
optimizer steps on gold solutions.

## v3-dpo

**Pairs:** 4,447 pairs from all 16,000 `dpo` prompts (the prompt cap was hit before the target of
6,000). The pair yield was 0.278, the same as the gate's mixed fraction. 3,988 pairs were used for
training and 459 held out.
- **Rejected answers:** 4,444 wrong and 3 invalid format.
- **Templates:** 37% `mul_sub_mul`, 22% `sub_mul_add`.

**Training:** 125 steps.

**Held-out pairs:** implicit-reward accuracy went from 0 (every margin exactly 0, since the policy started equal to
the reference) to 0.63, mean
β-margin +0.07. The policy's raw preference for the chosen response fell slightly (0.79 → 0.77).

**Val got worse,** most of all on *easy* templates that were rare in the pairs:

| Template | v2-sft | v3-dpo |
| --- | ---: | ---: |
| add_ab | 99.0% | 89.0% |
| sub_ab | 99.4% | 88.6% |
| x_plus_a_equals_b | 96.0% | 88.0% |
| paren_sum_times_diff | 83.2% | 71.2% |

Chosen and rejected traces share most of their tokens. Pushing the rejected sequence down
plausibly also lowers shared correct steps elsewhere. That is a hypothesis, not tested here.

## By template (val, greedy show-work)

| Template | v2-sft | v4-grpo | v2-extra-sft | v3-dpo |
| --- | ---: | ---: | ---: | ---: |
| add_ab | 99.0 | 96.4 | 100.0 | 89.0 |
| sub_ab | 99.4 | 99.0 | 100.0 | 88.6 |
| mul_ab | 99.6 | 99.2 | 99.8 | 99.2 |
| div_ab | 98.2 | 97.4 | 99.0 | 98.8 |
| ax_equals_b | 98.4 | 97.4 | 100.0 | 98.2 |
| x_plus_a_equals_b | 96.0 | 93.4 | 99.6 | 88.0 |
| add_mul_abc | 91.8 | 92.0 | 96.2 | 88.8 |
| ax_plus_b_equals_c | 98.8 | 96.4 | 99.6 | 92.6 |
| mul_sub_mul | 38.0 | 35.4 | **52.6** | 37.0 |
| sub_mul_add | 82.0 | 81.0 | 91.6 | 75.4 |
| ax_plus_b_equals_cx_plus_d | 98.8 | 97.4 | 99.8 | 93.4 |
| a_paren_x_plus_b_equals_c | 81.6 | 80.2 | 86.4 | 76.6 |
| a_times_paren_diff | 93.4 | 93.2 | 98.2 | 85.8 |
| paren_sum_times_diff | 83.2 | 79.4 | 92.4 | 71.2 |

**v2-extra-sft improves every template.** That includes +14.6 on `mul_sub_mul`. Under the SFT
format it now matches v1-math's plain-format accuracy (93.9%), which removes the cost SFT had
shown earlier.

## Failure checks (spec §7) for the negative GRPO result

| Check | Result |
| --- | --- |
| Baseline success | 0.70 sampled on the pool, so not near 0 or 1 |
| Mixed-group frequency | 0.27, above the gate but leaving about ¾ of groups without signal |
| Reward implementation | Same scorer as the evaluation; analytic sign/direction tests pass; 0 verifier errors |
| Clip fraction | ≈ 0, so updates were near on-policy and clipping did not bind |
| KL drift | Small (≤ 0.014 per token) |
| Truncation | ≈ 0 |
| Compute mismatch | GRPO's time is generation-dominated; the control got 9× the optimizer steps |

**Candidate causes, none isolated by an ablation:**
- A low LR (5e-6) with norm-clipped updates.
- Few informative groups per step (about 9 of 32).
- A small group size (4).
- Only 600 updates.

**Unchanged:** none of these changes the conclusion for this configuration.

## Spend and safety

| Item | Value |
| --- | --- |
| Pod lifetime | 18:31:01 → 20:17 UTC, ≈ 106 min, **≈ $1.75** at the listed rate (storage excluded) |
| Budget ledgers at the end of each run | v4-grpo $0.84; v2-extra-sft $1.04; v3-dpo $1.26; no stops |
| Watchdog | 22:01:01 UTC hard deadline armed first; a 45-minute self-delete was armed after the session; neither fired |
| Termination | Checkpoints and the rollout record verified locally, `/root/collected` set, pod deleted (204; pod list empty) |

**Incidents (idle billed time, about 50 minutes ≈ $0.85 in total):**
- **Setup.** The setup watcher's process check matched its own SSH command line, so it never saw
  setup exit. The pod sat idle from 18:38 to 19:10.
- **Session end.** The log-tail monitor delivered the completion line late. The session ended at
  19:54, but results were collected at 20:13.
- **Fix for both:** a future watcher should key on log markers and file timestamps, not `pgrep -f`.

## Artifacts

- **Checkpoints** (local, not tracked): `artifacts/p2-v4-grpo`, `artifacts/p2-v2-extra-sft` and
  `artifacts/p2-v3-dpo` link into `runs/pod_results/iabuazi1zbika0/`. All three are
  checksum-verified.
- **Rollout record:** `runs/pod_results/iabuazi1zbika0/runs/rl/v4-grpo/rollouts.jsonl`.
- **DPO pairs:** `.../v3-dpo/pairs.jsonl`.
- **Per-prompt eval rows:** `.../runs/rl/eval/*/rows.jsonl`.

## What this suggests next (not run)

- **H4.** Writing the confirmatory analysis plan and opening the held-out-template test suite is
  the only way to settle H4. On val, GRPO is already 5.5 points behind its control.
- **A second GRPO configuration.** If GRPO is worth another look, run it as a recorded protocol
  change, with a fresh approval. Candidates:
  - higher LR;
  - larger groups (8);
  - a pool reweighted on *sampled* rather than greedy accuracy;
  - dropping zero-variance groups from the batch so more of each batch carries signal.

  It would need the same time-matched control.
