# RL-stage protocol v1: gate, GRPO, equal-compute SFT control, DPO

Date: 2026-11-08. Recorded **before** any RL-stage run. All branches start from the protocol-v2
`v2-sft` checkpoint (`artifacts/p2-v2-sft`, val greedy show-work 89.9%). The frozen evaluation
suites are unchanged and the test suites stay sealed. Every result in the session report will be
on **val**.

## Training-only prompt pool (stream rl1)

`configs/data_stream_rl.yaml`: about 880 k new problems from the same generators and
in-distribution ranges, with every frozen evaluation problem excluded and an audit before use.

**Template weights.** v2-sft already solves most problems, so a uniform pool would give mostly
all-correct groups, which carry no GRPO signal. Each template's weight is proportional to
1 − p⁴ − (1 − p)⁴, the chance that a group of 4 samples mixes successes and failures. Here p is
v2-sft's greedy show-work val accuracy on that template. Each share is floored at 2%, then the
shares are renormalized.
- **Effect:** the expected mixed-group fraction is about 0.49, against about 0.25 uniform.
- **Val use:** this uses val accuracy per template to design the *training* distribution only. No
  selection on val outcomes of the branches follows from it.

**Partitions.** Records are split by a hash of their id (`microreason.rl.pool`), so the four uses
never share a problem:

| Partition | Share | Use |
| --- | ---: | --- |
| gate | 0.5% | reward-variance gate (512 prompts) |
| grpo | 25% | GRPO rollout prompts (9,600 used, no repeats) |
| dpo | 14.5% | DPO pair prompts |
| sft_extra | 60% | v2-extra-sft examples (gold solutions) |

## Reward-variance gate (spec §6, Stage 5)

v2-sft samples 4 completions for each of 512 `gate` prompts, with GRPO's exact sampling settings:
temperature 0.8, at most 128 new tokens, show-work format.

**Declared rule.** GRPO runs only if all three hold:
- the mixed-group fraction is at least 0.20;
- the malformed fraction (invalid format or truncated) is at most 0.10;
- pass@1 is between 0.05 and 0.95.

If the gate fails, the session stops before GRPO and reports the result. No RL spend goes into
"fixing" it.

## v4-grpo (`configs/grpo.yaml`, `src/microreason/rl/grpo.py`)

**Rollouts and reward**
- 300 steps × 32 prompts × 4 samples = 38,400 completions.
- Primary binary reward: 1 only for a finished response with exactly one valid answer block and
  the exact correct answer.
- Group-standardized advantages; a zero-variance group gets zero advantage.

**Objective**
- Clipped ratio, ε = 0.2. Old-policy log probabilities are recomputed by the generating weights.
- 2 minibatch updates per rollout step, so staleness is at most 1 update. Rollouts are never
  reused across steps.
- k3 KL to the frozen v2-sft, β = 0.04.
- Log probabilities use temperature-scaled logits.
- Per-token terms are averaged within each sequence, then across sequences. Prompt and padding
  tokens are masked.
- Truncated completions are kept in the loss with reward 0.

**Optimization**
- AdamW at LR 5e-6 (v2-sft's final LR), 10 warmup steps, then constant LR. No LR sweep.

**Records:** an immutable `rollouts.jsonl` (token IDs, old log probabilities, rewards, statuses,
generation settings) is kept, with its sha256 in the run manifest.

## v2-extra-sft: the equal-compute control (`configs/extra_sft.yaml`)

- **Recipe:** continue v2-sft with the unchanged SFT recipe (75/25 show-work/direct, loss on
  assistant tokens only, 64 examples per step, AdamW), at a constant LR of 5e-6.
- **Data:** the `sft_extra` partition, which has the same template mix as GRPO's prompts.
- **Compute match, declared now:** extra-SFT's training loop is stopped at GRPO's measured
  `train_wall_seconds`. That figure covers GRPO's rollouts, verification and updates, on the same
  pod and GPU.
  - Final evaluations are excluded for both branches.
  - The report will give GPU time, optimizer steps, examples and tokens for both, because the
    objectives spend compute differently.

## v3-dpo (`configs/dpo.yaml`, `src/microreason/rl/dpo.py`)

**Pairs**
- v2-sft samples 4 completions for each `dpo` prompt. Each mixed group gives one pair: its first
  correct completion is chosen and its first incorrect one is rejected, as judged by the exact
  verifier.
- Prompts are processed until there are 6,000 pairs, or until 16,000 prompts have been sampled.
- 10% of pairs are held out by prompt hash, for discrimination checks.

**Training:** DPO with β = 0.1 and v2-sft frozen as the reference; sequence log probabilities over
response tokens; LR 2e-6, inside the spec's range, with no sweep; 1 epoch.

## Evaluation and comparisons (val, paired)

Each of v2-sft (re-evaluated on the same pod), v4-grpo, v2-extra-sft and v3-dpo is evaluated:
- greedy in `sft-show-work-v1`, plus 4 samples at temperature 0.8 for pass@k;
- greedy in `sft-direct-v1`.

**Paired bootstrap comparisons** (greedy, both formats):

| Comparison | Bearing on |
| --- | --- |
| v4-grpo vs v2-sft | H4 |
| v4-grpo vs v2-extra-sft | H4 control |
| v2-extra-sft vs v2-sft | — |
| v3-dpo vs v2-sft | H3 |

**What val cannot settle.** H4's confirmatory test is defined on the held-out-template **test**
suite (spec §7), which stays sealed. These val results can show whether the branches move
in-distribution accuracy, but they **cannot** support or reject H4 as declared. The final
confirmatory analysis plan must be written before any test suite is opened.

## Budget

- **Approval:** the owner approved this work in chat on 2026-11-08, after an estimate of about
  $1–3.
- **Bounds** (`configs/budget_rl_v1.yaml`): one RunPod Secure pod (RTX 5090), a $5 cap, a
  3.5-hour limit, a watchdog with self-termination, and termination at the end.
