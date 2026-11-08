# RL-stage protocol v2: a stronger GRPO run and its time-matched SFT control

Date: 2026-11-08. Recorded **before** any rl-v2 run, after the v1 results (`reports/rl_v1.md`).
Every branch starts from the protocol-v2 `v2-sft` checkpoint (`artifacts/p2-v2-sft`, checkpoint
id `sha256:1ec3ff35…2a904`). The RL prompt pool, its partitions and the reward are unchanged from
v1 (`reports/rl_protocol_v1.md`). Every result in this protocol is on **val**. The confirmatory
test evaluation that follows in the same session is declared separately, in
`reports/confirmatory_plan_v1.md`.

## Why a second configuration

v1 GRPO left val accuracy 1.5 points below v2-sft and 5.5 points below its time-matched SFT
control. Its training signals showed a run that barely moved:
- Training reward was flat (about 0.69).
- About 73% of groups had zero reward variance, so only about 9 of 32 prompts per step carried
  any signal.
- Updates were norm-clipped at LR 5e-6, and KL to the parent stayed at 0.01 or below.
- 82% of training time went to uncached generation, so the time-matched control got about 9× as
  many optimizer steps.

v1 did not isolate a cause. rl-v2 changes several things at once to give GRPO a fair attempt. It
is **not** an ablation and cannot attribute any change to one factor.

## Changes from v1 (all declared now)

| | v1 | rl-v2 |
| --- | --- | --- |
| Rollout sampler | uncached, full recompute per token | KV cache (`model.KVCache`); evaluation unchanged |
| Group size | 4 | 8 |
| Groups trained on | all 32 sampled, 73% with zero advantage | only mixed groups; 32 per step (dynamic sampling) |
| LR (constant after 10 warmup steps) | 5e-6 | 1e-5 **and** 3e-5, two runs, one selected |
| Generation batch | 128 | 256 |
| Steps × trained completions | 300 × 128 = 38,400 | 300 × 256 = 76,800 |

**Unchanged:** clip ε 0.2, k3 KL to frozen v2-sft with β 0.04, temperature 0.8, at most
128 new tokens, 2 minibatch updates per step (staleness at most 1 update), sequence-mean
normalization, binary exact-answer reward, seed 45, bf16, and the immutable rollout record.

**Dynamic sampling.** Each step draws chunks of 64 new prompts from the `grpo` partition, in a
fixed seeded order with no repeats, and samples 8 completions per prompt from the current policy.
Groups with mixed outcomes are kept until 32 are collected or 8 chunks have been drawn; surplus
mixed groups are dropped. Every sampled completion is still written to `rollouts.jsonl`. Those not
trained on are marked `used_in_loss: false`, with no old log probabilities and advantage 0.
- **Cost of the filter:** dropped groups had zero advantage anyway, so they lose only their KL
  term.
- **Empty steps:** a step that finds no mixed group makes no update and is logged.

**KV-cache parity check (gate before any rl-v2 rollout).** On the pod, before any training,
`microreason rl kv-check` runs v2-sft on 512 `gate`-partition prompts (training-only) in the
show-work format. The session stops before any GRPO run unless both of these hold:
- the fp32 greedy completions of the cached and uncached generators are identical for at least
  99% of prompts;
- the bf16 sampled pass@1 (4 samples, temperature 0.8, the same seed) of the two samplers differs
  by at most 0.02.

CPU parity tests on small models (`tests/test_eval.py`) already show identical greedy and seeded
sampled outputs, and logits agree to within 1e-4 of the logit scale. The GRPO loss never uses
the sampler's probabilities: old-policy log probabilities are recomputed by teacher forcing.

**Gate.** The v1 reward-variance gate (same parent, pool and temperature, group size 4) passed
and is not rerun.

## Hyperparameter search and selection (exploratory)

Two GRPO runs are made, at LR 1e-5 and 3e-5 (`configs/grpo_v2_lr1e-5.yaml` and
`configs/grpo_v2_lr3e-5.yaml`). Each is evaluated on val exactly as in v1.

**Selection rule:** the run with the higher val greedy show-work exact accuracy is selected; a tie
goes to the lower LR (`scripts/rl_v2_decide.py select`). The selected run is called
**v4-grpo-r2** in every later artifact. This is a search over two settings on val, so the val
numbers for v4-grpo-r2 carry selection bias and are labelled exploratory. The test suites play
no part in selection.

## v2-extra-sft-r2: the equal-compute control

- **Recipe and data:** the unchanged `configs/extra_sft.yaml`, the same recipe, data partition
  and seed as v1's control.
- **Time limit:** stopped at the selected run's measured `train_wall_seconds`, on the same pod.
  That figure covers rollouts including dropped groups, verification and updates; final
  evaluations are excluded for both branches.
- **Search cost not matched:** the unselected GRPO run's time is not added to the control's
  budget. The report gives it separately.
- **Reported for both branches:** GPU time, optimizer steps, examples and tokens.

## Deviations from the spec's initial GRPO settings (§6 Stage 5)

| Spec | Here | Reason |
| --- | --- | --- |
| Group of 4 completions | 8 | More mixed groups and a better group baseline |
| Initial LR 1e-6 | 1e-5 / 3e-5 | v1 at 5e-6 barely moved the policy |
| 20,000–40,000 completions (planning range) | 76,800 trained, more sampled | Dynamic sampling and faster rollouts |
| Zero-variance groups logged, contribute no signal | Logged and recorded, not trained on | Also drops their KL term |

These are training-recipe choices made on val before any test result exists. The spec allows
them as long as they are reported.

## Evaluation (val, paired)

- **Checkpoints:** v2-sft (re-evaluated on this pod), both GRPO v2 runs and v2-extra-sft-r2.
- **Decoding:** greedy show-work plus 4 samples at temperature 0.8, and greedy direct.
- **Paired bootstraps:** each GRPO run vs v2-sft; v2-extra-sft-r2 vs v2-sft; v4-grpo-r2 vs
  v2-extra-sft-r2.

## Budget

- **Approval:** the owner approved rl-v2 and the confirmatory test evaluation in chat on
  2026-11-08, after estimates of about $1–2 and $0.50–1.
- **Bounds** (`configs/budget_rl_v2.yaml`): one RunPod Secure pod, a $4 cap, a 3.5-hour limit,
  the watchdog with self-termination, and termination at the end.
