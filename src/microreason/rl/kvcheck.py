"""On-pod check that KV-cache rollouts match the uncached sampler on the real checkpoint.

Protocol rl-v2 uses the cached sampler for GRPO rollouts only. Before any rollout, the session
runs this check on v2-sft with 512 `gate`-partition prompts (training-only), show-work format:

- **fp32 greedy:** the cached and uncached generators must produce identical completions for at
  least `min_fp32_greedy_agreement` of prompts. Only floating-point rounding may separate them,
  and that flips a token only at a near-tie.
- **bf16 sampled (the rollout precision):** both samplers draw 4 completions per prompt at
  temperature 0.8 with the same seed. Their pass@1 must differ by at most `max_pass_at_1_gap`.
  Rounding differs between the paths, so individual samples can diverge; their distribution
  must not.

The report also gives both samplers' wall time on the bf16 sampled pass, the speedup GRPO v2
relies on. The session stops before any training if the check fails.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import torch

from microreason.rl.rollout import ROLLOUT_FORMAT, autocast, sample_groups

MIN_FP32_GREEDY_AGREEMENT = 0.99
MAX_PASS_AT_1_GAP = 0.02


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def run_kv_check(resolved, checkpoint: str, out: str | Path, device: str,
                 records: list[dict] | None = None, prompts: int = 512) -> dict:  # fmt: skip
    from microreason.checkpoint import load_checkpoint
    from microreason.eval.generate import eval_prompt_ids, generate_all
    from microreason.rl.pool import load_partition
    from microreason.tokenizer import MicroTokenizer

    cfg = resolved.stage
    s = cfg.sampling
    if records is None:
        records, _ = load_partition(cfg.pool_manifest, "gate", prompts)
    tok = MicroTokenizer.load(cfg.tokenizer)
    model, _, _ = load_checkpoint(checkpoint, device)
    model.eval()
    dev = torch.device(device)
    ids = [eval_prompt_ids(tok, r["prompt"], ROLLOUT_FORMAT) for r in records]
    common = dict(max_new_tokens=s.max_new_tokens, eos_id=tok.eos_id, pad_id=tok.pad_id)

    with torch.no_grad():
        g_plain = generate_all(model, ids, batch_size=s.generation_batch_size, **common)
        g_cache = generate_all(model, ids, batch_size=s.generation_batch_size, use_cache=True,
                               **common)  # fmt: skip
    agree = sum(a.ids == b.ids for a, b in zip(g_plain, g_cache, strict=True)) / len(ids)

    sampled = {}
    for name, use_cache in (("uncached", False), ("cached", True)):
        gen = torch.Generator(device=dev).manual_seed(cfg.seed + 2000)
        _sync(dev)
        t0 = time.time()
        groups = sample_groups(model, tok, records, group_size=4, temperature=s.temperature,
                               max_new_tokens=s.max_new_tokens, generator=gen,
                               batch_size=s.generation_batch_size, precision=cfg.precision,
                               use_cache=use_cache)  # fmt: skip
        _sync(dev)
        comps = [c for g in groups for c in g]
        sampled[name] = {
            "seconds": round(time.time() - t0, 3),
            "pass_at_1": sum(c.reward for c in comps) / len(comps),
            "mean_tokens": sum(len(c.ids) for c in comps) / len(comps),
        }
    with torch.no_grad(), autocast(dev, cfg.precision):
        b_plain = generate_all(model, ids, batch_size=s.generation_batch_size, **common)
        b_cache = generate_all(model, ids, batch_size=s.generation_batch_size, use_cache=True,
                               **common)  # fmt: skip
    bf16_agree = sum(a.ids == b.ids for a, b in zip(b_plain, b_cache, strict=True)) / len(ids)

    gap = abs(sampled["cached"]["pass_at_1"] - sampled["uncached"]["pass_at_1"])
    failures = []
    if agree < MIN_FP32_GREEDY_AGREEMENT:
        failures.append(f"fp32 greedy agreement {agree:.4f} < {MIN_FP32_GREEDY_AGREEMENT}")
    if gap > MAX_PASS_AT_1_GAP:
        failures.append(f"bf16 sampled pass@1 gap {gap:.4f} > {MAX_PASS_AT_1_GAP}")
    report = {
        "checkpoint": str(checkpoint),
        "prompts": len(ids),
        "partition": "gate",
        "rule": {
            "min_fp32_greedy_agreement": MIN_FP32_GREEDY_AGREEMENT,
            "max_pass_at_1_gap": MAX_PASS_AT_1_GAP,
        },  # fmt: skip
        "fp32_greedy_agreement": agree,
        "bf16_greedy_agreement": bf16_agree,
        "bf16_sampled": sampled,
        "pass_at_1_gap": gap,
        "speedup": round(
            sampled["uncached"]["seconds"] / max(sampled["cached"]["seconds"], 1e-9), 2
        ),  # fmt: skip
        "passed": not failures,
        "failures": failures,
    }
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report
