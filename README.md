# MicroReason

A small (~42.7M parameter) decoder language model, trained from random initialization, to test whether
arithmetic and basic algebra skills can be learned through a staged pipeline (base pretraining → math
mid-training → SFT → DPO / GRPO) within a bounded compute budget.

The full design is in [MICROREASON_MASTER_SPEC.md](MICROREASON_MASTER_SPEC.md).

## Status

**Stage 1 trained: v0-base and v1-math** (RTX 4090, ≈$0.64). On val, math mid-training lifts exact
accuracy from 25.8% to 48.9% (paired +23.2 pts, 95% CI [+22.1, +24.3]); see `reports/stage1_v1.md`.
**v2-sft trained** (≈$0.44): val show-work 50.0%, direct 21.8%, 100% valid format, pass@4 60.5%;
see `reports/stage2_v1.md`.
**Protocol v2 (scaled: 1B base + 200M mid tokens, ≈$2.19):** val accuracy v0-base 88.1%, v1-math
93.9%, v2-sft show-work 89.9% (pass@4 93.6%); see `reports/scale_v2.md` and `reports/protocol_v2.md`.

**RL stage v1 (≈$1.75, val only):** v4-grpo 88.4% (−1.5 vs v2-sft), equal-compute v2-extra-sft
93.9% (+4.1), v3-dpo 84.5% (−5.4); the reward-variance gate passed (mixed groups 0.28). See
`reports/rl_v1.md` and `reports/rl_protocol_v1.md`.
**RL stage v2 + confirmatory test (≈$2.31):** GRPO v2 (KV-cache rollouts, mixed-group sampling,
group 8) val 88.1% (−1.8) vs time-matched v2-extra-sft-r2 94.4% (+4.5). **H4 not supported** on
the held-out-template test: every model is near 0% there (v4-grpo-r2 0.07%, v2-sft 0.17%,
v2-extra-sft-r2 0.20%), and near 0% on number extrapolation; `test_id` 87.8 / 90.2 / 94.3%.
See `reports/rl_v2.md`, `reports/rl_protocol_v2.md` and `reports/confirmatory_plan_v1.md`.
The test suites have now been used once; any later test evaluation is post-hoc.
**Closeout:** H1 and H2 supported (val), H3 and H4 not supported; see `reports/closeout_v1.md`.
Out-of-distribution failures (tokenizer-limited number copying, template substitution) are
analysed in `reports/failure_analysis_v1.md`.
**Protocol v3 Phase B (≈$2.76):** digit tokenizer + tree-generated data. v2-sft solves 70.3% of
held-out-*shape* val problems (v1 held-out templates: 0.17%), as well as same-size training
shapes; accuracy is limited by depth (95% at 1 operator, ~37% at 5). H1 +5.2, H2 +78.1 on val.
Capability gate passed; see `reports/v3_phase_b.md` and `reports/protocol_v3.md`.
**Protocol v3 Phase C (≈$0.63, two attempts):** the RL reward-variance gate failed twice (mixed
groups 0.182 at T 0.8, 0.195 at T 1.0; rule ≥ 0.20). v2-sft's outcomes are near all-or-nothing per
prompt. No GRPO ran; the v3 test suites stay sealed; H4 undecided. See `reports/v3_phase_c_attempt2.md`.
Math training data: a 1.47M-problem training-only stream (≈70M tokens) excludes every frozen
evaluation problem; see `reports/math_stream_v1.md`. No real training stage or evaluation has run yet. Every number in the spec is a target or estimate, not a measured result.

| Phase | Scope | Status |
| --- | --- | --- |
| 1 | Package, typed config schema, logging, CLI skeleton, tests | Done (local tests green) |
| 2 | Data generators, splits, manifests, answer verifier | Done; eval suites frozen (`data/manifests/math_eval_frozen_v1.json`) |
| 3 | Tokenizer and model | Done; see `reports/phase3_inspection_v1.md` |
| 4 | Common training loop, resume, smoke tests | Done; see `reports/phase4_training_loop_v1.md` |
| 5 | Authorized GPU pilot | Done; see `reports/phase5_pilot_v1.md` |
| 6 | Base pretraining and math mid-training | Done (val only; test sealed) |
| 7 | SFT (v2-sft) | Done (val only); DPO, GRPO and SFT control not started |
| 8 | Branch comparisons, sealed test evaluation, final report | Not started |

## Setup

Requires [uv](https://docs.astral.sh/uv/). The project pins Python 3.13 via `.python-version`.

```bash
uv sync
uv run pytest
uv run microreason --help
```

## What works today

```bash
uv run microreason config validate --config configs/pretrain.yaml   # validate and print resolved config + hash
uv run microreason model inspect --config configs/model_43m.yaml    # instantiate; count + memory
uv run microreason tokenizer train                                  # train tokenizer (train splits only)
uv run microreason tokenizer inspect                                # diagnostics + real token counts
```

Data pipeline:

```bash
uv run microreason data fetch-sources   # download pinned Cosmopedia files (~1.2 GB), verify sha256
uv run microreason data split-text      # hash-split general text into train/val
uv run microreason data generate        # build math splits + manifests, scan corpus for overlap
uv run microreason data audit           # re-verify every record, check cross-split leakage
uv run microreason data generate-stream # ~1.47M-problem training-only math stream (2 disjoint shards)
uv run microreason data audit-stream    # stream vs frozen suites: full overlap check + sampled re-verify
uv run microreason data freeze          # lock evaluation suite hashes (only after review)
```

Generated JSONL lives in `data/generated/` (not tracked; rebuilt deterministically). Manifests with
hashes and counts are tracked in `data/manifests/`; the audit report is in `reports/`.

Training (local, $0):

```bash
uv run microreason data tokenize                                    # pre-tokenize train/val datasets
uv run microreason train --config configs/smoke.yaml --run-dir runs/smoke_v1_a
uv run microreason train --config configs/smoke.yaml --run-dir runs/smoke_v1_b \
    --resume runs/smoke_v1_a/checkpoints/step_0000100                  # exact resume
```

Evaluation (frozen suites only; test suites are sealed unless `--allow-test` is passed):

```bash
uv run microreason eval --suite val --checkpoint runs/<run>/checkpoints/step_N \
    --format plain-v1 --out runs/eval/<name> [--samples 4]       # rows.jsonl + summary.json
uv run microreason compare --a runs/eval/<a> --b runs/eval/<b>    # paired bootstrap, b - a
```

Prompt formats: `plain-v1` for base and mid-trained checkpoints (continues to `Answer: x`),
`sft-show-work-v1` and `sft-direct-v1` for later stages. Math mid-training follows an easy-to-hard
curriculum (`configs/midtrain.yaml`), with uniform sampling kept as an ablation
(`configs/midtrain_uniform.yaml`).

Runs and checkpoints live in `runs/` (not tracked). Heavy jobs (training, evaluation of real
checkpoints) run on rented GPUs, not locally. SFT, DPO and GRPO stages and `serve` are
placeholders that exit with an error naming the phase that will implement them.

## Spending

The config schema defaults to `paid_compute_allowed: false` and `max_spend_usd: 0`. Enabling paid
compute requires a provider, dollar cap, runtime cap, and shutdown rule in the config **and** a separate
explicit approval from the project owner. Nothing in this repository authorizes spending on its own.

Paid runs additionally need an hourly rate, a pod id and the billing start time. A budget guard
stops training before the tighter of the runtime and dollar ceilings (minus a reserve), saves a
checkpoint, and keeps an atomic ledger in `budget_ledger.json`. An on-pod watchdog terminates the
pod at the hard deadline. `configs/pilot_43m.yaml` stays invalid until its approval fields are
filled in.
