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
    # 0 = serial generation with shared de-duplication (the v1 semantics, kept so stream v1
    # reproduces). >0 = one process per template writing its own part, merged in fixed template
    # order with global de-duplication (deterministic; used for the large v2 stream).
    parallel_workers: int = Field(default=0, ge=0)
    tokenized_prefix: str = "math_stream"


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


def _verify_file(path: str | Path, digest: str) -> None:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        while chunk := f.read(1 << 24):
            h.update(chunk)
    if h.hexdigest() != digest:
        raise ValueError(f"{path} does not match its manifest sha256")


def iter_stream_dicts(manifest: dict):
    """Yield raw record dicts, streaming, after verifying the whole file's hash first."""
    _verify_file(manifest["file"], manifest["sha256"])
    with Path(manifest["file"]).open("r", encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)


def iter_stream(manifest: dict):
    """Yield validated records (streaming, hash-verified)."""
    from microreason.data.schema import ProblemRecord

    for d in iter_stream_dicts(manifest):
        yield ProblemRecord.model_validate(d)


def audit_stream(cfg: StreamConfig, sample_fraction: float = 0.02, seed: int = 0) -> dict:
    """Full leakage and duplicate check against the frozen suites, plus schema validation and
    independent re-verification of a random sample."""
    from microreason.data.checks import independent_check, step_check
    from microreason.data.schema import ProblemRecord

    manifest = json.loads(Path(cfg.manifest_path).read_text())
    fps, prompts, _ = frozen_eval_keys(cfg.eval_manifest_dir, cfg.eval_version)
    rng = random.Random(seed)
    seen_fp: set[str] = set()
    failures: Counter = Counter()
    n = checked = 0
    for r in iter_stream_dicts(manifest):
        n += 1
        if r["split"] != SPLIT or r["number_range"] != "in_distribution":
            failures["wrong_split_or_range"] += 1
        if TEMPLATES_BY_ID[r["template_id"]].heldout:
            failures["heldout_template"] += 1
        fp = r["semantic_fingerprint"]
        if fp in fps:
            failures["fingerprint_in_eval"] += 1
        if normalize_prompt(r["prompt"]) in prompts:
            failures["prompt_in_eval"] += 1
        if fp in seen_fp:
            failures["duplicate_in_stream"] += 1
        seen_fp.add(fp)
        if rng.random() < sample_fraction:
            checked += 1
            rec = ProblemRecord.model_validate(r)
            reason = independent_check(
                rec.verifier_spec.kind, rec.verifier_spec.text, rec.prompt, rec.final_answer
            ) or step_check(rec.verifier_spec.kind, rec.solution_steps, rec.final_answer)
            if reason:
                failures[f"recheck:{reason}"] += 1
    return {
        "records": n,
        "rechecked_sample": checked,
        "failures": dict(failures),
        "passed": n == manifest["record_count"] and not failures,
    }


# ----------------------------------------------------------------------------- parallel (v2)

_EVAL_KEYS: tuple[set[str], set[str]] = (set(), set())


def _init_worker(fps: set[str], prompts: set[str]) -> None:
    global _EVAL_KEYS
    _EVAL_KEYS = (fps, prompts)


def _generate_part(cfg_json: str, tid: str, target: int, part: str) -> tuple[str, int, dict]:
    """Generate one template's problems straight to a part file (no large in-memory lists)."""
    import random as _random

    from microreason.data.checks import independent_check, step_check
    from microreason.data.generators import GenerationReject, Sampler
    from microreason.verify.expr import fmt_value

    cfg = StreamConfig.model_validate_json(cfg_json)
    t = TEMPLATES_BY_ID[tid]
    seen_fp = set(_EVAL_KEYS[0])
    seen_prompt = set(_EVAL_KEYS[1])
    rejections: Counter = Counter()
    n = 0
    limit = target * cfg.max_attempts_factor + 1000
    with open(part, "wb") as f:
        for idx in range(limit):
            seed = build.derive_seed(cfg.root_seed, tid, SPLIT, idx)
            try:
                p = t.fn(Sampler(_random.Random(seed), "in_distribution"))
            except GenerationReject as e:
                rejections[f"sampler:{e}"] += 1
                continue
            fp = build.fingerprint(p)
            npr = normalize_prompt(p.prompt)
            if fp in seen_fp or npr in seen_prompt:
                rejections["duplicate"] += 1
                continue
            answer = fmt_value(p.answer)
            reason = independent_check(p.kind, p.math_text, p.prompt, answer) or step_check(
                p.kind, p.steps, answer
            )
            if reason:
                rejections[reason] += 1
                continue
            seen_fp.add(fp)
            seen_prompt.add(npr)
            cand = build.Candidate(t, SPLIT, seed, p, fp, "")
            f.write(build._record_line(build._to_record(cand, n)).encode("utf-8"))
            n += 1
            if n == target:
                break
    return tid, n, dict(rejections)


def build_stream_parallel(cfg: StreamConfig) -> dict:
    from concurrent.futures import ProcessPoolExecutor

    unknown = set(cfg.targets) - set(TEMPLATES_BY_ID)
    heldout = {t for t in cfg.targets if TEMPLATES_BY_ID.get(t) and TEMPLATES_BY_ID[t].heldout}
    if unknown or heldout:
        raise ValueError(
            f"bad stream templates: unknown={sorted(unknown)} held-out={sorted(heldout)}"
        )
    fps, prompts, frozen = frozen_eval_keys(cfg.eval_manifest_dir, cfg.eval_version)
    out = Path(cfg.out_path)
    parts_dir = out.parent / "parts"
    parts_dir.mkdir(parents=True, exist_ok=True)
    cfg_json = cfg.model_dump_json()
    # Largest targets first so the slowest parts start earliest.
    jobs = sorted(cfg.targets.items(), key=lambda kv: -kv[1])
    results: dict[str, tuple[int, dict]] = {}
    with ProcessPoolExecutor(cfg.parallel_workers, initializer=_init_worker,
                             initargs=(fps, prompts)) as pool:  # fmt: skip
        futures = [pool.submit(_generate_part, cfg_json, tid, n, str(parts_dir / f"{tid}.jsonl"))
                   for tid, n in jobs]  # fmt: skip
        for fut in futures:
            tid, n, rej = fut.result()
            results[tid] = (n, rej)
            log.info("stream part %-28s %8d / %8d", tid, n, cfg.targets[tid])

    # Merge in config template order with global de-duplication (cross-template collisions).
    seen_fp: set[str] = set()
    seen_prompt: set[str] = set()
    h = hashlib.sha256()
    shards: Counter = Counter()
    by_level: Counter = Counter()
    kept: Counter = Counter()
    cross_dupes: Counter = Counter()
    tmp = out.with_suffix(".jsonl.partial")
    with tmp.open("wb") as fo:
        for tid in cfg.targets:
            with (parts_dir / f"{tid}.jsonl").open("rb") as fi:
                for line in fi:
                    r = json.loads(line)
                    npr = normalize_prompt(r["prompt"])
                    if r["semantic_fingerprint"] in seen_fp or npr in seen_prompt:
                        cross_dupes[tid] += 1
                        continue
                    seen_fp.add(r["semantic_fingerprint"])
                    seen_prompt.add(npr)
                    fo.write(line)
                    h.update(line)
                    kept[tid] += 1
                    shards[shard_of(r["id"], cfg.shard_a_fraction)] += 1
                    by_level[str(r["difficulty"])] += 1
    tmp.replace(out)
    for p in parts_dir.glob("*.jsonl"):
        p.unlink()
    rejections: Counter = Counter()
    for _, rej in results.values():
        rejections.update(rej)
    manifest = {
        "schema_version": 1,
        "dataset": "math",
        "version": cfg.version,
        "split": SPLIT,
        "file": str(out),
        "sha256": h.hexdigest(),
        "record_count": sum(kept.values()),
        "generator_version": GENERATOR_VERSION,
        "generation_mode": f"parallel per template ({cfg.parallel_workers} workers), merged in "
        "config order with global de-duplication",
        "root_seed": cfg.root_seed,
        "stream_config_hash": stream_config_hash(cfg),
        "targets": dict(cfg.targets),
        "achieved": dict(kept),
        "generated_before_merge": {t: n for t, (n, _) in results.items()},
        "cross_template_duplicates_dropped": dict(cross_dupes),
        "capacity_limited": sorted(
            t for t in cfg.targets if results[t][0] < cfg.targets[t]
        ),  # fmt: skip
        "by_difficulty": dict(sorted(by_level.items())),
        "shards": {
            "rule": f"sha256(id) bucket < {cfg.shard_a_fraction} -> a, else b",
            **dict(sorted(shards.items())),
        },  # fmt: skip
        "excluded_frozen_suites": {s: frozen[s]["sha256"] for s in EVAL_SPLITS},
        "excluded_fingerprints": len(fps),
        "build_rejections": dict(sorted(rejections.items())),
        "code_revision": build._code_revision(),
    }
    Path(cfg.manifest_path).write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def generate_stream(cfg: StreamConfig) -> dict:
    return build_stream_parallel(cfg) if cfg.parallel_workers else build_stream(cfg)
