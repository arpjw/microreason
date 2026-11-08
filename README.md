# MicroReason

A small (~42.7M parameter) decoder language model, trained from random initialization, to test whether
arithmetic and basic algebra skills can be learned through a staged pipeline (base pretraining → math
mid-training → SFT → DPO / GRPO) within a bounded compute budget.

The full design is in [MICROREASON_MASTER_SPEC.md](MICROREASON_MASTER_SPEC.md).

## Status

**Stage 1 trained: v0-base and v1-math** (RTX 4090, ≈$0.64). On val, math mid-training lifts exact
accuracy from 25.8% to 48.9% (paired +23.2 pts, 95% CI [+22.1, +24.3]); see `reports/stage1_v1.md`.
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
| 7–8 | SFT, DPO/GRPO branches, final evaluation, report | Not started |

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
