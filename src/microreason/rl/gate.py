"""Reward-variance gate before GRPO (spec §6 Stage 5).

Samples groups from the RL training distribution (the `gate` partition of stream rl1, disjoint
from GRPO's own prompts) with GRPO's exact sampling settings, and applies the decision rule
declared in the GRPO config before any GRPO run. Only training-only prompts are used.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

from microreason.rl.config import GRPOConfig, ResolvedRL
from microreason.rl.rollout import group_stats, sample_groups


def decide(overall: dict, cfg: GRPOConfig) -> tuple[bool, list[str]]:
    r = cfg.gate
    failures = []
    if overall["mixed_group_fraction"] < r.min_mixed_group_fraction:
        failures.append(f"mixed_group_fraction {overall['mixed_group_fraction']:.3f} < "
                        f"{r.min_mixed_group_fraction}")  # fmt: skip
    if overall["malformed_fraction"] > r.max_malformed_fraction:
        failures.append(f"malformed_fraction {overall['malformed_fraction']:.3f} > "
                        f"{r.max_malformed_fraction}")  # fmt: skip
    if not r.min_pass_at_1 <= overall["pass_at_1"] <= r.max_pass_at_1:
        failures.append(f"pass_at_1 {overall['pass_at_1']:.3f} outside "
                        f"[{r.min_pass_at_1}, {r.max_pass_at_1}]")  # fmt: skip
    return not failures, failures


def run_gate(resolved: ResolvedRL, checkpoint: str, out: str | Path, device: str,
             records: list[dict] | None = None) -> dict:  # fmt: skip
    from microreason.checkpoint import load_checkpoint
    from microreason.rl.pool import PARTITION_RULE, load_partition
    from microreason.tokenizer import MicroTokenizer

    cfg = resolved.stage
    if not isinstance(cfg, GRPOConfig):
        raise ValueError("the gate uses the GRPO config's sampling settings and rule")
    pool_sha = "in-memory (tests)"
    if records is None:
        records, m = load_partition(cfg.pool_manifest, "gate", cfg.gate.prompts)
        pool_sha = m["sha256"]
    tok = MicroTokenizer.load(cfg.tokenizer)
    model, _, _ = load_checkpoint(checkpoint, device)
    s = cfg.sampling
    gen = torch.Generator(device=device).manual_seed(cfg.seed + 1000)
    groups = sample_groups(model, tok, records, group_size=s.group_size,
                           temperature=s.temperature, max_new_tokens=s.max_new_tokens,
                           generator=gen, batch_size=s.generation_batch_size,
                           precision=cfg.precision)  # fmt: skip
    stats = group_stats(records, groups)
    passed, failures = decide(stats["overall"], cfg)
    report = {
        "checkpoint": str(checkpoint),
        "pool": {
            "manifest": cfg.pool_manifest,
            "sha256": pool_sha,
            "partition": "gate",
            "partition_rule": PARTITION_RULE,
            "prompts": len(records),
        },  # fmt: skip
        "sampling": s.model_dump(),
        "rule": cfg.gate.model_dump(),
        "passed": passed,
        "failures": failures,
        **stats,
    }
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report
