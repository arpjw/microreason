# MicroReason

A small (~42.7M parameter) decoder language model, trained from random initialization, to test whether
arithmetic and basic algebra skills can be learned through a staged pipeline (base pretraining → math
mid-training → SFT → DPO / GRPO) within a bounded compute budget.

The full design is in [MICROREASON_MASTER_SPEC.md](MICROREASON_MASTER_SPEC.md).

## Status

**Phase 1 of 8: repository scaffold only.** No tokenizer, model, dataset, training run, checkpoint,
or evaluation exists yet. Every number in the spec is a target or estimate, not a measured result.

| Phase | Scope | Status |
| --- | --- | --- |
| 1 | Package, typed config schema, logging, CLI skeleton, tests | Done (local tests green) |
| 2 | Data generators, splits, manifests, answer verifier | Not started |
| 3 | Tokenizer and model | Not started |
| 4 | Common training loop, resume, smoke tests | Not started |
| 5 | Authorized GPU pilot | Not started (needs explicit approval) |
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
uv run microreason model inspect --config configs/model_43m.yaml    # analytic parameter count
```

All other subcommands (`data`, `tokenizer`, `train`, `eval`, `compare`, `serve`) are placeholders that
exit with an error naming the phase that will implement them.

## Spending

The config schema defaults to `paid_compute_allowed: false` and `max_spend_usd: 0`. Enabling paid
compute requires a provider, dollar cap, runtime cap, and shutdown rule in the config **and** a separate
explicit approval from the project owner. Nothing in this repository authorizes spending on its own.
