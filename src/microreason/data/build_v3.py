"""Build, audit and freeze data v3 (generator 3.0, reports/protocol_v3.md).

Same leakage discipline as v1 (`data.build`):

1. **Evaluation candidates first.** Candidates are generated for val, test_id, test_extrapolation
   and the two held-out shape suites, with one shared de-duplication state. They are scanned
   against the general-text corpus and trimmed to their targets.
2. **Then the SFT training split.** It rejects anything already used by an evaluation candidate.
3. **The pretraining and mid-training stream is built separately** (`build_stream_v3`). It is
   generated in parallel parts and merged in a fixed order, with global de-duplication against
   the frozen suites.

**Shape constraints by split:**

| Split | Shapes |
| --- | --- |
| train, stream, val, test_id, test_extrapolation | train-visible only |
| val_heldout_shape | held-out-val only |
| test_heldout_shape | held-out-test only |

**Seeds** are `sha256(root_seed | generator | split | index)`, so every record regenerates
from its own seed.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, PositiveInt

from microreason.data.build import _code_revision
from microreason.data.checks import independent_check, step_check
from microreason.data.generators import GenerationReject, Problem
from microreason.data.schema import V3_EVAL_SPLITS, ProblemRecord, VerifierSpec
from microreason.data.split import (
    ContaminationScanner,
    answer_type,
    equation_fingerprint,
    expression_fingerprint,
    iter_corpus_texts,
    normalize_math,
    normalize_prompt,
)
from microreason.data.trees import (
    GENERATOR_VERSION_V3,
    coverage_report,
    difficulty,
    family,
    sample_problem,
    shape_set,
)
from microreason.logging_utils import get_logger
from microreason.verify.expr import fmt_value

log = get_logger("microreason.data.build_v3")

REQUIRED_SET: dict[str, str] = {
    "train": "train",
    "train_stream": "train",
    "val": "train",
    "test_id": "train",
    "test_extrapolation": "train",
    "val_heldout_shape": "heldout_val",
    "test_heldout_shape": "heldout_test",
}


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class CountsV3(_Strict):
    train: PositiveInt = 100_000
    val: PositiveInt = 7000
    test_id: PositiveInt = 7000
    test_extrapolation: PositiveInt = 7000
    val_heldout_shape: PositiveInt = 6000
    test_heldout_shape: PositiveInt = 6000


class CorpusScanV3(_Strict):
    enabled: bool = True
    sources_config: str = "configs/data_sources.yaml"
    raw_dir: str = "data/raw"


class StreamV3(_Strict):
    records: PositiveInt = 8_000_000
    shard_a_fraction: float = Field(default=0.6, gt=0, lt=1)
    part_size: PositiveInt = 250_000  # seed indices per parallel part
    workers: PositiveInt = 12
    out_path: str = "data/generated/math_v3/train_stream/train_stream.jsonl"
    manifest_path: str = "data/manifests/math_train_stream_v3.json"
    tokenized_prefix: str = "math_stream"


class RLPoolV3(_Strict):
    """Training-only RL pool (protocol v3 Phase C): GRPO prompts, the reward-variance gate and the
    equal-compute SFT control's examples, split by `rl.pool` partitions. A separate seed stream
    (its own root seed), excluding every frozen v3 evaluation record.

    Family weights (declared before any RL run): a candidate of family f is kept with probability
    w_f / max(w), w_f = 1 - p_f^k - (1 - p_f)^k, the chance that a group of k samples from v2-sft
    mixes successes and failures, with p_f v2-sft's greedy show-work val accuracy for f
    (reports/v3_phase_b.md). The keep/drop draw is a hash of the record seed, so it is
    deterministic."""

    root_seed: int = 20261111
    records: PositiveInt = 2_000_000
    group_size: PositiveInt = 8
    family_accuracy: dict[str, float] = Field(default_factory=dict)
    # Protocol v4: keep probabilities declared directly (overrides family_accuracy). All 1.0
    # gives the generator's natural family mix.
    family_keep: dict[str, float] | None = None
    # Record ids are f"{id_prefix}_{n:08d}"; partitions are a hash of the id, so pools that
    # coexist (protocol v4's SFT and RL pools) must use different prefixes.
    id_prefix: str = "v3_train_rl"
    purpose: str = "rl_pool"
    part_size: PositiveInt = 250_000
    workers: PositiveInt = 12
    out_path: str = "data/generated/math_v3/train_rl/train_rl.jsonl"
    manifest_path: str = "data/manifests/math_train_rl_v3.json"


class DataV3Config(_Strict):
    dataset: str = "math"
    version: str = "v3"
    root_seed: int
    counts: CountsV3 = CountsV3()
    overgenerate: float = Field(default=1.2, ge=1.0, le=5.0)
    max_attempts_factor: PositiveInt = 80
    corpus_scan: CorpusScanV3 = CorpusScanV3()
    stream: StreamV3 = StreamV3()
    rl_pool: RLPoolV3 = RLPoolV3()
    out_dir: str = "data/generated/math_v3"
    manifest_dir: str = "data/manifests"


def load_v3_config(path: str | Path) -> DataV3Config:
    return DataV3Config.model_validate(yaml.safe_load(Path(path).read_text()))


def config_hash(cfg: DataV3Config) -> str:
    blob = json.dumps(cfg.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
    return "sha256:" + hashlib.sha256(blob.encode()).hexdigest()


def derive_seed(root_seed: int, split: str, index: int) -> int:
    key = f"{root_seed}|{GENERATOR_VERSION_V3}|{split}|{index}"
    return int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "big")


def number_range_for(split: str) -> str:
    return "extrapolation" if split == "test_extrapolation" else "in_distribution"


def fingerprint(p: Problem) -> str:
    if p.kind == "expression":
        return expression_fingerprint(p.tree)
    return equation_fingerprint(p.lhs, p.rhs)


@dataclass
class Candidate:
    split: str
    seed: int
    problem: Problem
    shape: str
    fingerprint: str
    math_key: str


@dataclass
class State:
    seen_fp: set[str] = field(default_factory=set)
    seen_prompt: set[str] = field(default_factory=set)
    rejections: Counter = field(default_factory=Counter)


def generate_one(root_seed: int, split: str, index: int) -> tuple[Candidate | None, str]:
    """Generate and check one candidate. Returns (candidate or None, rejection reason)."""
    return generate_from_seed(derive_seed(root_seed, split, index), split)


def generate_from_seed(seed: int, split: str) -> tuple[Candidate | None, str]:
    try:
        p, shape = sample_problem(random.Random(seed), number_range_for(split))
    except GenerationReject as e:
        return None, f"sampler:{e}"
    if shape_set(shape) != REQUIRED_SET[split]:
        return None, "shape_not_in_split_set"
    answer = fmt_value(p.answer)
    reason = independent_check(p.kind, p.math_text, p.prompt, answer) or step_check(
        p.kind, p.steps, answer
    )
    if reason:
        return None, reason
    return Candidate(split, seed, p, shape, fingerprint(p), normalize_math(p.math_text)), ""


def _generate(cfg: DataV3Config, split: str, n: int, state: State) -> list[Candidate]:
    out: list[Candidate] = []
    limit = n * cfg.max_attempts_factor + 1000
    for idx in range(limit):
        c, why = generate_one(cfg.root_seed, split, idx)
        if c is None:
            state.rejections[why] += 1
            continue
        npr = normalize_prompt(c.problem.prompt)
        if c.fingerprint in state.seen_fp or npr in state.seen_prompt:
            state.rejections["duplicate"] += 1
            continue
        state.seen_fp.add(c.fingerprint)
        state.seen_prompt.add(npr)
        out.append(c)
        if len(out) == n:
            return out
    raise RuntimeError(f"{split}: only {len(out)} of {n} problems after {limit} attempts")


def to_record(c: Candidate, index: int) -> ProblemRecord:
    p = c.problem
    return ProblemRecord(
        id=f"v3_{c.split}_{index:08d}",
        family=family(c.shape),
        template_id=c.shape,
        difficulty=difficulty(c.shape),
        prompt=p.prompt,
        solution_steps=p.steps,
        answer_type=answer_type(p.answer),
        final_answer=fmt_value(p.answer),
        verifier_spec=VerifierSpec(
            kind=p.kind, text=p.math_text, variable="x" if p.kind == "linear_equation" else None
        ),
        generator_version=GENERATOR_VERSION_V3,
        seed=c.seed,
        split=c.split,  # type: ignore[arg-type]
        number_range=number_range_for(c.split),  # type: ignore[arg-type]
        params=p.params,
        semantic_fingerprint=c.fingerprint,
    )


def record_line(r: ProblemRecord) -> str:
    return json.dumps(r.model_dump(mode="json"), sort_keys=True, ensure_ascii=False) + "\n"


def manifest_path(manifest_dir: str | Path, version: str, split: str) -> Path:
    return Path(manifest_dir) / f"math_{split}_{version}.json"


def _counts(records: list[ProblemRecord]) -> dict:
    return {
        "by_family": dict(sorted(Counter(r.family for r in records).items())),
        "by_difficulty": dict(sorted(Counter(str(r.difficulty) for r in records).items())),
        "by_answer_type": dict(sorted(Counter(r.answer_type for r in records).items())),
        "by_kind": dict(sorted(Counter(r.verifier_spec.kind for r in records).items())),
        "distinct_shapes": len({r.template_id for r in records}),
    }


def build_v3(cfg: DataV3Config) -> dict:
    """Generate the evaluation suites and the SFT training split. Returns the records by split."""
    cov = coverage_report()
    if not cov["passed"]:
        raise RuntimeError("shape coverage check failed: " + "; ".join(cov["failures"]))
    state = State()
    cands: dict[str, list[Candidate]] = {}
    for split in V3_EVAL_SPLITS:
        target = getattr(cfg.counts, split)
        cands[split] = _generate(cfg, split, math.ceil(target * cfg.overgenerate), state)
        log.info("candidates %-20s %d", split, len(cands[split]))

    contaminated: set[str] = set()
    scan: dict = {"enabled": cfg.corpus_scan.enabled}
    if cfg.corpus_scan.enabled:
        scanner = ContaminationScanner({c.math_key for cs in cands.values() for c in cs})
        files: set[str] = set()
        for file_sha, text in iter_corpus_texts(cfg.corpus_scan.sources_config,
                                                cfg.corpus_scan.raw_dir):  # fmt: skip
            files.add(file_sha)
            scanner.scan(text)
        contaminated = scanner.found
        scan.update(corpus_files=sorted(files), documents_scanned=scanner.docs_scanned,
                    candidate_keys=scanner.n_keys, contaminated_keys=len(contaminated))  # fmt: skip

    records: dict[str, list[ProblemRecord]] = {}
    dropped = Counter()
    for split, cs in cands.items():
        target = getattr(cfg.counts, split)
        clean = [c for c in cs if c.math_key not in contaminated]
        dropped[split] = len(cs) - len(clean)
        if len(clean) < target:
            raise RuntimeError(f"{split}: {len(clean)} clean candidates for {target}")
        records[split] = [to_record(c, i) for i, c in enumerate(clean[:target])]
    scan["contaminated_candidates_dropped"] = dict(dropped)
    train = _generate(cfg, "train", cfg.counts.train, state)
    records["train"] = [to_record(c, i) for i, c in enumerate(train)]
    return {"records": records, "rejections": dict(state.rejections), "corpus_scan": scan,
            "coverage": cov}  # fmt: skip


def guard_against_frozen_v3(cfg: DataV3Config, result: dict) -> None:
    """Refuse a build whose evaluation suites differ from already-frozen ones (before writing)."""
    path = Path(cfg.manifest_dir) / f"math_eval_frozen_{cfg.version}.json"
    if not path.exists():
        return
    frozen = json.loads(path.read_text())["suites"]
    for split in V3_EVAL_SPLITS:
        data = "".join(record_line(r) for r in result["records"][split]).encode("utf-8")
        if hashlib.sha256(data).hexdigest() != frozen[split]["sha256"]:
            raise ValueError(f"{split} would change the frozen v3 suite; refusing to write")


def write_v3(cfg: DataV3Config, result: dict) -> list[Path]:
    out_dir = Path(cfg.out_dir)
    (out_dir / "eval").mkdir(parents=True, exist_ok=True)
    Path(cfg.manifest_dir).mkdir(parents=True, exist_ok=True)
    paths = []
    for split, recs in result["records"].items():
        f = out_dir / ("train" if split == "train" else "eval") / f"{split}.jsonl"
        f.parent.mkdir(parents=True, exist_ok=True)
        data = "".join(record_line(r) for r in recs).encode("utf-8")
        f.write_bytes(data)
        m = {
            "schema_version": 1, "dataset": "math", "version": cfg.version, "split": split,
            "file": str(f), "sha256": hashlib.sha256(data).hexdigest(),
            "record_count": len(recs), "generator_version": GENERATOR_VERSION_V3,
            "root_seed": cfg.root_seed, "data_config_hash": config_hash(cfg),
            "required_shape_set": REQUIRED_SET[split], "counts": _counts(recs),
            "corpus_scan": result["corpus_scan"], "build_rejections": result["rejections"],
            "coverage": {k: v for k, v in result["coverage"].items() if k != "failures"},
            "code_revision": _code_revision(),
        }  # fmt: skip
        p = manifest_path(cfg.manifest_dir, cfg.version, split)
        p.write_text(json.dumps(m, indent=2, sort_keys=True) + "\n")
        paths.append(p)
    return paths


def load_split_v3(manifest: dict) -> list[ProblemRecord]:
    data = Path(manifest["file"]).read_bytes()
    if hashlib.sha256(data).hexdigest() != manifest["sha256"]:
        raise ValueError(f"{manifest['file']} does not match its manifest hash")
    return [ProblemRecord.model_validate_json(x) for x in data.decode("utf-8").splitlines()]


def audit_v3(manifest_dir: str | Path, version: str = "v3", regenerate: int = 2000) -> dict:
    """Cross-split checks and a regeneration check. Every eval record is verified; the first
    `regenerate` records of each split are regenerated from their seed and must match."""
    failures: list[str] = []
    seen_fp: dict[str, str] = {}
    seen_prompt: dict[str, str] = {}
    splits = {}
    for split in (*V3_EVAL_SPLITS, "train"):
        m = json.loads(manifest_path(manifest_dir, version, split).read_text())
        recs = load_split_v3(m)
        bad = Counter()
        for i, r in enumerate(recs):
            if shape_set(r.template_id) != REQUIRED_SET[split]:
                bad["shape_set"] += 1
            for key, seen in ((r.semantic_fingerprint, seen_fp),
                              (normalize_prompt(r.prompt), seen_prompt)):  # fmt: skip
                if key in seen and seen[key] != split:
                    bad[f"overlap_with_{seen[key]}"] += 1
                elif key in seen:
                    bad["duplicate"] += 1
                seen[key] = split
            if split != "train" or i < regenerate:
                why = independent_check(r.verifier_spec.kind, r.verifier_spec.text, r.prompt,
                                        r.final_answer) or step_check(
                    r.verifier_spec.kind, r.solution_steps, r.final_answer)  # fmt: skip
                if why:
                    bad[why] += 1
            if i < regenerate:
                c, _ = generate_from_seed(r.seed, split)
                if c is None:
                    bad["regeneration_mismatch"] += 1
                else:
                    regen = to_record(c, i).model_dump()
                    regen["id"] = r.id  # ids are assigned at write time (stream renumbers)
                    if regen != r.model_dump():
                        bad["regeneration_mismatch"] += 1
        splits[split] = {"record_count": len(recs), "sha256": m["sha256"], "bad": dict(bad)}
        if bad:
            failures.append(f"{split}: {dict(bad)}")
    return {"passed": not failures, "failures": failures, "splits": splits}


def freeze_v3(manifest_dir: str | Path, version: str = "v3") -> Path:
    report = audit_v3(manifest_dir, version)
    if not report["passed"]:
        raise ValueError("audit failed; refusing to freeze:\n" + "\n".join(report["failures"]))
    suites = {s: {"sha256": report["splits"][s]["sha256"],
                  "record_count": report["splits"][s]["record_count"]}
              for s in V3_EVAL_SPLITS}  # fmt: skip
    frozen = {"dataset": "math", "version": version, "suites": suites}
    path = Path(manifest_dir) / f"math_eval_frozen_{version}.json"
    if path.exists():
        if json.loads(path.read_text())["suites"] != suites:
            raise ValueError(f"{path} already freezes different suites; refusing to overwrite")
        return path
    path.write_text(json.dumps(frozen, indent=2, sort_keys=True) + "\n")
    return path


# ----------------------------------------------------------------------------- stream


def family_keep_probability(pool: RLPoolV3) -> dict[str, float]:
    k = pool.group_size
    w = {f: 1 - p**k - (1 - p) ** k for f, p in pool.family_accuracy.items()}
    top = max(w.values())
    return {f: round(v / top, 6) for f, v in w.items()}


def _keep(seed: int, prob: float) -> bool:
    u = int(hashlib.sha256(f"rl-pool-keep:{seed}".encode()).hexdigest()[:12], 16) / 16**12
    return u < prob


def _stream_part(args: tuple) -> tuple[int, str, dict]:
    root_seed, part, part_size, tmp_dir, keep = args
    from microreason.data.build_v3 import _EVAL_KEYS  # set in the worker initializer

    fps, prompts = _EVAL_KEYS
    out = Path(tmp_dir) / f"part_{part:05d}.jsonl"
    rej = Counter()
    seen: set[str] = set()
    with out.open("w", encoding="utf-8") as f:
        for idx in range(part * part_size, (part + 1) * part_size):
            c, why = generate_one(root_seed, "train_stream", idx)
            if c is None:
                rej[why] += 1
                continue
            if keep is not None:
                fam = family(c.shape)
                if fam not in keep:
                    raise ValueError(f"no declared accuracy for family {fam}")
                if not _keep(c.seed, keep[fam]):
                    rej["family_weight_drop"] += 1
                    continue
            npr = normalize_prompt(c.problem.prompt)
            if c.fingerprint in fps or npr in prompts:
                rej["frozen_eval_overlap"] += 1
                continue
            if c.fingerprint in seen:
                rej["duplicate"] += 1
                continue
            seen.add(c.fingerprint)
            f.write(json.dumps({"i": idx, "c": c.shape, "fp": c.fingerprint, "np": npr,
                                "rec": record_line(to_record(c, 0))}) + "\n")  # fmt: skip
    return part, str(out), dict(rej)


_EVAL_KEYS: tuple[set[str], set[str]] = (set(), set())


def _init_stream_worker(fps: set[str], prompts: set[str]) -> None:
    global _EVAL_KEYS
    _EVAL_KEYS = (fps, prompts)


def build_stream_v3(cfg: DataV3Config) -> dict:
    """Training-only stream for base and mid-training, excluding every frozen v3 eval record."""
    sc = cfg.stream
    return _build_stream(cfg, cfg.root_seed, sc.records, sc.part_size, sc.workers, sc.out_path,
                         sc.manifest_path, sc.shard_a_fraction, keep=None,
                         id_prefix="v3_train_stream")  # fmt: skip


def build_rl_pool_v3(cfg: DataV3Config) -> dict:
    """Training-only RL pool for Phase C (see RLPoolV3)."""
    rp = cfg.rl_pool
    if rp.family_keep is not None:
        if not all(0 < v <= 1 for v in rp.family_keep.values()):
            raise ValueError("rl_pool.family_keep probabilities must be in (0, 1]")
        keep = dict(rp.family_keep)
    elif rp.family_accuracy:
        keep = family_keep_probability(rp)
    else:
        raise ValueError("declare rl_pool.family_accuracy or rl_pool.family_keep before building")
    m = _build_stream(cfg, rp.root_seed, rp.records, rp.part_size, rp.workers, rp.out_path,
                      rp.manifest_path, 0.5, keep=keep, id_prefix=rp.id_prefix)  # fmt: skip
    from microreason.rl.pool import PARTITION_RULE, partition_of

    parts = Counter()
    fams = Counter()
    for line in open(m["file"], encoding="utf-8"):
        r = json.loads(line)
        parts[partition_of(r["id"])] += 1
        fams[r["family"]] += 1
    m.update(purpose=rp.purpose, family_accuracy=dict(rp.family_accuracy),
             group_size=rp.group_size, family_keep_probability=keep,
             partitions={"rule": PARTITION_RULE, **dict(sorted(parts.items()))},
             by_family=dict(sorted(fams.items())))  # fmt: skip
    Path(rp.manifest_path).write_text(json.dumps(m, indent=2, sort_keys=True) + "\n")
    return m


def _build_stream(cfg: DataV3Config, root_seed: int, records: int, part_size: int, workers: int,
                  out_path: str, out_manifest: str, shard_a: float, keep,
                  id_prefix: str) -> dict:  # fmt: skip
    from microreason.data.stream import shard_of

    frozen_path = Path(cfg.manifest_dir) / f"math_eval_frozen_{cfg.version}.json"
    if not frozen_path.exists():
        raise ValueError("v3 evaluation suites are not frozen; refusing to build a stream")
    frozen = json.loads(frozen_path.read_text())["suites"]
    fps, prompts = set(), set()
    for split in V3_EVAL_SPLITS:
        m = json.loads(manifest_path(cfg.manifest_dir, cfg.version, split).read_text())
        if m["sha256"] != frozen[split]["sha256"]:
            raise ValueError(f"{split} does not match its frozen hash")
        for r in load_split_v3(m):
            fps.add(r.semantic_fingerprint)
            prompts.add(normalize_prompt(r.prompt))
    out = Path(out_path)
    tmp_dir = out.parent / "parts"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    # conservative: only the first `records` records in part order are kept, so extra parts
    # change nothing but generation time
    yield_rate = 0.6 if keep is None else 0.6 * max(0.2, sum(keep.values()) / len(keep)) * 0.7
    n_parts = math.ceil(records / (yield_rate * part_size)) + 1
    jobs = [(root_seed, p, part_size, str(tmp_dir), keep) for p in range(n_parts)]
    results = {}
    with ProcessPoolExecutor(workers, initializer=_init_stream_worker,
                             initargs=(fps, prompts)) as ex:  # fmt: skip
        for part, path, rej in ex.map(_stream_part, jobs):
            results[part] = (path, rej)
            log.info("stream part %d done", part)
    rejections = Counter()
    for _, rej in results.values():
        rejections.update(rej)
    seen_fp: set[bytes] = set()
    seen_np: set[bytes] = set()
    h = hashlib.sha256()
    shards, by_level, shapes = Counter(), Counter(), set()
    n = 0
    tmp = out.with_suffix(".jsonl.partial")
    with tmp.open("wb") as f:
        for part in range(n_parts):
            for line in open(results[part][0], encoding="utf-8"):
                if n == records:
                    break
                row = json.loads(line)
                kf = hashlib.sha256(row["fp"].encode()).digest()[:16]
                kp = hashlib.sha256(row["np"].encode()).digest()[:16]
                if kf in seen_fp or kp in seen_np:
                    rejections["duplicate_across_parts"] += 1
                    continue
                seen_fp.add(kf)
                seen_np.add(kp)
                rec = json.loads(row["rec"])
                rec["id"] = f"{id_prefix}_{n:08d}"
                line_b = (json.dumps(rec, sort_keys=True, ensure_ascii=False) + "\n").encode()
                f.write(line_b)
                h.update(line_b)
                shards[shard_of(rec["id"], shard_a)] += 1
                by_level[str(rec["difficulty"])] += 1
                shapes.add(row["c"])
                n += 1
    if n < records:
        raise RuntimeError(f"stream: only {n} of {records} records; raise part count")
    tmp.replace(out)
    manifest = {
        "schema_version": 1, "dataset": "math", "version": cfg.version,
        "split": "train_stream", "file": str(out), "sha256": h.hexdigest(),
        "record_count": n, "generator_version": GENERATOR_VERSION_V3,
        "root_seed": root_seed, "data_config_hash": config_hash(cfg),
        "by_difficulty": dict(sorted(by_level.items())), "distinct_shapes": len(shapes),
        "shards": {"rule": f"sha256(id) bucket < {shard_a} -> a, else b",
                   **dict(sorted(shards.items()))},
        "excluded_frozen_suites": {s: frozen[s]["sha256"] for s in V3_EVAL_SPLITS},
        "excluded_fingerprints": len(fps), "build_rejections": dict(sorted(rejections.items())),
        "code_revision": _code_revision(),
    }  # fmt: skip
    Path(out_manifest).write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest
