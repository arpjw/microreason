# MicroReason

A small (~42.7M parameter) decoder language model, trained from random initialization, to test whether
arithmetic and basic algebra skills can be learned through a staged pipeline (base pretraining → math
mid-training → SFT → DPO / GRPO) within a bounded compute budget.

The full design is in [MICROREASON_MASTER_SPEC.md](MICROREASON_MASTER_SPEC.md).

## Status

**Phase 5 of 8 done: GPU pilot complete** (RTX 5090, ≈252k tok/s for the 43M model, ≈$1.03 total).
No real training stage or evaluation has run yet. Every number in the spec is a target or estimate, not a measured result.

| Phase | Scope | Status |
| --- | --- | --- |
| 1 | Package, typed config schema, logging, CLI skeleton, tests | Done (local tests green) |
| 2 | Data generators, splits, manifests, answer verifier | Done; eval suites frozen (`data/manifests/math_eval_frozen_v1.json`) |
| 3 | Tokenizer and model | Done; see `reports/phase3_inspection_v1.md` |
| 4 | Common training loop, resume, smoke tests | Done; see `reports/phase4_training_loop_v1.md` |
| 5 | Authorized GPU pilot | Done; see `reports/phase5_pilot_v1.md` |
| 6–8 | Training stages, branches, evaluation, report | Not started |

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

Runs and checkpoints live in `runs/` (not tracked). SFT, DPO and GRPO stages and the
`eval`, `compare` and `serve` commands are placeholders that exit with an error naming the
phase that will implement them.

## Spending

The config schema defaults to `paid_compute_allowed: false` and `max_spend_usd: 0`. Enabling paid
compute requires a provider, dollar cap, runtime cap, and shutdown rule in the config **and** a separate
explicit approval from the project owner. Nothing in this repository authorizes spending on its own.

Paid runs additionally need an hourly rate, a pod id and the billing start time. A budget guard
stops training before the tighter of the runtime and dollar ceilings (minus a reserve), saves a
checkpoint, and keeps an atomic ledger in `budget_ledger.json`. An on-pod watchdog terminates the
pod at the hard deadline. `configs/pilot_43m.yaml` stays invalid until its approval fields are
filled in.
