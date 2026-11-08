"""Run a checkpoint on a frozen suite; write per-prompt rows and an aggregate summary (spec §7).

Only frozen suites can be evaluated, and their hashes are re-verified. Test suites are sealed:
evaluating one needs allow_test=True, so development work only ever touches `val`.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import torch

from microreason.data import build
from microreason.data.schema import EVAL_SPLITS
from microreason.eval.generate import FORMATS, eval_prompt_ids, generate_all
from microreason.eval.score import (
    paired_bootstrap,
    score_completion,
    summarize_greedy,
    summarize_samples,
)
from microreason.logging_utils import get_logger
from microreason.tokenizer import MicroTokenizer

log = get_logger("microreason.eval")
EVAL_PROTOCOL = "eval-v1"


class SealedSuiteError(RuntimeError):
    pass


def load_frozen_suite(split: str, manifest_dir: str = "data/manifests", version: str = "v1"):
    if split not in EVAL_SPLITS:
        raise ValueError(f"{split!r} is not an evaluation suite")
    frozen = build.frozen_suites(manifest_dir, version)
    if frozen is None:
        raise ValueError("evaluation suites are not frozen")
    m = json.loads(build.manifest_path(manifest_dir, version, split).read_text())
    if m["sha256"] != frozen[split]["sha256"]:
        raise ValueError(f"{split} does not match its frozen hash")
    return [r.model_dump(mode="json") for r in build.load_split(m)], m["sha256"]


def _rows(records, gens, tok, fmt, suite_hash, ckpt_id, decode, latency):
    rows = []
    for rec, g in zip(records, gens, strict=True):
        completion = tok.decode(g.ids)
        rows.append({
            "prompt_id": rec["id"],
            "family": rec["family"],
            "template_id": rec["template_id"],
            "difficulty": rec["difficulty"],
            "suite_sha256": suite_hash,
            "checkpoint_id": ckpt_id,
            "prompt_format": fmt,
            "decode": decode,
            "raw_completion": completion,
            "expected_answer": rec["final_answer"],
            "generated_tokens": len(g.ids),
            "hit_eos": g.hit_eos,
            "latency_s_amortized": latency,
            **score_completion(rec, completion, g.hit_eos, fmt),
        })  # fmt: skip
    return rows


def run_eval(
    split: str,
    checkpoint: str | Path,
    fmt: str,
    out_dir: str | Path,
    *,
    tokenizer_dir: str = "artifacts/tokenizer/v1",
    manifest_dir: str = "data/manifests",
    allow_test: bool = False,
    samples: int = 0,
    temperature: float = 0.8,
    seed: int = 0,
    max_new_tokens: int = 128,
    batch_size: int = 64,
    limit: int | None = None,
    device: str = "cpu",
) -> dict:
    from microreason.checkpoint import MANIFEST, load_checkpoint

    if fmt not in FORMATS:
        raise ValueError(f"unknown prompt format {fmt!r}")
    if split.startswith("test") and not allow_test:
        raise SealedSuiteError(
            f"{split} is sealed; pass allow_test only for predeclared final runs"
        )
    out = Path(out_dir)
    if (out / "summary.json").exists():
        raise FileExistsError(f"{out} already holds an evaluation; refusing to overwrite")

    records, suite_hash = load_frozen_suite(split, manifest_dir)
    if limit is not None:
        records = records[:limit]
    tok = MicroTokenizer.load(tokenizer_dir)
    model, _, ckpt_manifest = load_checkpoint(checkpoint, device)
    if ckpt_manifest.get("tokenizer_sha256") not in (None, tok.sha256):
        raise ValueError("checkpoint was trained with a different tokenizer")
    ckpt_id = "sha256:" + hashlib.sha256((Path(checkpoint) / MANIFEST).read_bytes()).hexdigest()
    model.eval()
    prompts = [eval_prompt_ids(tok, r["prompt"], fmt) for r in records]
    common = dict(max_new_tokens=max_new_tokens, eos_id=tok.eos_id, pad_id=tok.pad_id)

    t0 = time.time()
    greedy = generate_all(model, prompts, batch_size=batch_size, **common)
    dt = time.time() - t0
    rows = _rows(records, greedy, tok, fmt, suite_hash, ckpt_id, "greedy", dt / len(records))
    summary = {
        "protocol": EVAL_PROTOCOL,
        "suite": split,
        "suite_sha256": suite_hash,
        "limited_to": limit,
        "checkpoint": str(checkpoint),
        "checkpoint_id": ckpt_id,
        "checkpoint_step": ckpt_manifest.get("step"),
        "prompt_format": fmt,
        "device": device,
        "decode": {"greedy": {"max_new_tokens": max_new_tokens}},
        "greedy": summarize_greedy(rows),
        "greedy_seconds": round(dt, 3),
    }
    if samples:
        sample_rows = []
        for s in range(samples):
            g = torch.Generator(device=device).manual_seed(seed + s)
            gens = generate_all(model, prompts, batch_size=batch_size, temperature=temperature,
                                generator=g, **common)  # fmt: skip
            sample_rows += _rows(records, gens, tok, fmt, suite_hash, ckpt_id,
                                 f"sample:{s}", None)  # fmt: skip
        rows += sample_rows
        summary["decode"]["sampled"] = {"temperature": temperature, "samples": samples,
                                        "seed": seed, "max_new_tokens": max_new_tokens}  # fmt: skip
        summary["sampled"] = summarize_samples(sample_rows, samples)

    out.mkdir(parents=True, exist_ok=True)
    with (out / "rows.jsonl").open("w") as f:
        for r in rows:
            f.write(json.dumps(r, sort_keys=True) + "\n")
    (out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def compare_runs(a_dir: str | Path, b_dir: str | Path, resamples: int = 10_000, seed: int = 0):
    def load(d):
        s = json.loads((Path(d) / "summary.json").read_text())
        rows = [json.loads(x) for x in (Path(d) / "rows.jsonl").read_text().splitlines()]
        greedy = {r["prompt_id"]: r["status"] == "correct" for r in rows if r["decode"] == "greedy"}
        return s, greedy

    sa, a = load(a_dir)
    sb, b = load(b_dir)
    # The comparison uses greedy rows only, so only the greedy decode settings must match;
    # either run may additionally carry sampled pass@k rows.
    for key in ("suite_sha256", "prompt_format", "limited_to"):
        if sa.get(key) != sb.get(key):
            raise ValueError(f"runs differ in {key}; not a valid paired comparison")
    if sa["decode"].get("greedy") != sb["decode"].get("greedy"):
        raise ValueError("runs differ in greedy decode settings; not a valid paired comparison")
    result = paired_bootstrap(a, b, resamples, seed)
    result.update(a=str(a_dir), b=str(b_dir), suite=sa["suite"], prompt_format=sa["prompt_format"])
    return result
