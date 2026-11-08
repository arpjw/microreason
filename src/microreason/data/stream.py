"""Large training-only math stream for base pretraining and math mid-training.

The v1 train split (56k problems, 2.26 M tokens) is sized for SFT. Pretraining and mid-training
plan for about 70 M math tokens, which would repeat that split about 31 times. This module
generates a separate `train_stream` split with the same generators and in-distribution ranges:

- Every frozen evaluation problem is excluded by semantic fingerprint and normalized prompt.
  The frozen hashes are verified before generation starts.
- Capacity-limited templates (small single-operation spaces) contribute every unique problem
  they have. The others get per-template targets in configs/data_stream.yaml.
- Every record passes the independent SymPy check and the step check at build time.
- Records are assigned to disjoint shards by a hash of their id: shard `a` feeds base
  pretraining and shard `b` mid-training, so neither stage repeats the other's problems.

It may overlap the v1 SFT train split; both are training data.
"""

from __future__ import annotations

import hashlib
import json
import random
from collections import Counter
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, PositiveInt

from microreason.data import build
from microreason.data.generators import GENERATOR_VERSION, TEMPLATES_BY_ID
from microreason.data.schema import EVAL_SPLITS
from microreason.data.split import normalize_prompt
from microreason.logging_utils import get_logger

log = get_logger("microreason.data.stream")
SPLIT = "train_stream"


class StreamConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    version: str = "v1"
    root_seed: int
    max_attempts_factor: PositiveInt = 20
    shard_a_fraction: float = Field(gt=0, lt=1)
    targets: dict[str, PositiveInt]
    eval_manifest_dir: str = "data/manifests"
    eval_version: str = "v1"
    out_path: str = "data/generated/math_v1/train_stream/train_stream.jsonl"
    manifest_path: str = "data/manifests/math_train_stream_v1.json"


def load_stream_config(path: str | Path) -> StreamConfig:
    with Path(path).open("r", encoding="utf-8") as f:
        return StreamConfig.model_validate(yaml.safe_load(f))


def stream_config_hash(cfg: StreamConfig) -> str:
    blob = json.dumps(cfg.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
    return "sha256:" + hashlib.sha256(blob.encode()).hexdigest()


def shard_of(record_id: str, fraction_a: float) -> str:
    bucket = int(hashlib.sha256(record_id.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    return "a" if bucket < fraction_a else "b"


def frozen_eval_keys(manifest_dir: str, version: str) -> tuple[set[str], set[str], dict]:
    """Fingerprints and normalized prompts of every frozen evaluation record, hash-verified."""
    frozen = build.frozen_suites(manifest_dir, version)
    if frozen is None:
        raise ValueError("evaluation suites are not frozen; refusing to build a training stream")
    fps: set[str] = set()
    prompts: set[str] = set()
    for split in EVAL_SPLITS:
        m = json.loads(build.manifest_path(manifest_dir, version, split).read_text())
        if m["sha256"] != frozen[split]["sha256"]:
            raise ValueError(f"{split} on disk does not match the frozen suite hash")
        for r in build.load_split(m):  # also re-verifies the file hash
            fps.add(r.semantic_fingerprint)
            prompts.add(normalize_prompt(r.prompt))
    return fps, prompts, frozen


def build_stream(cfg: StreamConfig) -> dict:
    unknown = set(cfg.targets) - set(TEMPLATES_BY_ID)
    heldout = {t for t in cfg.targets if TEMPLATES_BY_ID.get(t) and TEMPLATES_BY_ID[t].heldout}
    if unknown or heldout:
        raise ValueError(
            f"bad stream templates: unknown={sorted(unknown)} held-out={sorted(heldout)}"
        )

    fps, prompts, frozen = frozen_eval_keys(cfg.eval_manifest_dir, cfg.eval_version)
    state = build.BuildState(seen_fp=set(fps), seen_prompt=set(prompts))
    out = Path(cfg.out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".jsonl.partial")
    h = hashlib.sha256()
    achieved: dict[str, int] = {}
    shards: Counter = Counter()
    by_level: Counter = Counter()
    with tmp.open("wb") as f:
        for tid, target in cfg.targets.items():
            t = TEMPLATES_BY_ID[tid]
            cands = build._generate(cfg, SPLIT, t, target, state, allow_partial=True)
            achieved[tid] = len(cands)
            for i, c in enumerate(cands):
                rec = build._to_record(c, i)
                line = build._record_line(rec).encode("utf-8")
                f.write(line)
                h.update(line)
                shards[shard_of(rec.id, cfg.shard_a_fraction)] += 1
                by_level[str(t.difficulty)] += 1
            log.info("stream %-28s %7d / %7d", tid, len(cands), target)
    tmp.replace(out)
    manifest = {
        "schema_version": 1,
        "dataset": "math",
        "version": cfg.version,
        "split": SPLIT,
        "file": str(out),
        "sha256": h.hexdigest(),
        "record_count": sum(achieved.values()),
        "generator_version": GENERATOR_VERSION,
        "root_seed": cfg.root_seed,
        "stream_config_hash": stream_config_hash(cfg),
        "targets": dict(cfg.targets),
        "achieved": achieved,
        "capacity_limited": sorted(t for t, n in achieved.items() if n < cfg.targets[t]),
        "by_difficulty": dict(sorted(by_level.items())),
        "shards": {
            "rule": f"sha256(id) bucket < {cfg.shard_a_fraction} -> a, else b",
            **dict(sorted(shards.items())),
        },  # fmt: skip
        "excluded_frozen_suites": {s: frozen[s]["sha256"] for s in EVAL_SPLITS},
        "excluded_fingerprints": len(fps),
        "build_rejections": dict(sorted(state.rejections.items())),
        "code_revision": build._code_revision(),
    }
    Path(cfg.manifest_path).write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def iter_stream(manifest: dict):
    """Yield records from the stream file after verifying its hash."""
    from microreason.data.schema import ProblemRecord

    data = Path(manifest["file"]).read_bytes()
    if hashlib.sha256(data).hexdigest() != manifest["sha256"]:
        raise ValueError(f"{manifest['file']} does not match its manifest sha256")
    for line in data.decode("utf-8").splitlines():
        yield ProblemRecord.model_validate_json(line)


def audit_stream(cfg: StreamConfig, sample_fraction: float = 0.02, seed: int = 0) -> dict:
    """Full leakage check against the frozen suites, plus re-verification of a random sample."""
    from microreason.data.checks import independent_check, step_check

    manifest = json.loads(Path(cfg.manifest_path).read_text())
    fps, prompts, _ = frozen_eval_keys(cfg.eval_manifest_dir, cfg.eval_version)
    rng = random.Random(seed)
    seen_fp: set[str] = set()
    failures: Counter = Counter()
    n = checked = 0
    for r in iter_stream(manifest):
        n += 1
        if r.split != SPLIT or r.number_range != "in_distribution":
            failures["wrong_split_or_range"] += 1
        if TEMPLATES_BY_ID[r.template_id].heldout:
            failures["heldout_template"] += 1
        if r.semantic_fingerprint in fps:
            failures["fingerprint_in_eval"] += 1
        if normalize_prompt(r.prompt) in prompts:
            failures["prompt_in_eval"] += 1
        if r.semantic_fingerprint in seen_fp:
            failures["duplicate_in_stream"] += 1
        seen_fp.add(r.semantic_fingerprint)
        if rng.random() < sample_fraction:
            checked += 1
            reason = independent_check(
                r.verifier_spec.kind, r.verifier_spec.text, r.prompt, r.final_answer
            ) or step_check(r.verifier_spec.kind, r.solution_steps, r.final_answer)
            if reason:
                failures[f"recheck:{reason}"] += 1
    return {
        "records": n,
        "rechecked_sample": checked,
        "failures": dict(failures),
        "passed": n == manifest["record_count"] and not failures,
    }
