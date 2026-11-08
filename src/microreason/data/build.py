"""Build, audit and freeze the generated math dataset (spec §4, §7).

Build order matters for leakage control: evaluation candidates are generated first (held-out
template, extrapolation, in-distribution test, validation), scanned against the general-text
corpus, and trimmed to their target sizes. Training data is generated last and rejects anything
whose fingerprint or normalized prompt was already used by any evaluation candidate.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import subprocess
from collections import Counter
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path

import yaml

from microreason.data.checks import independent_check, step_check
from microreason.data.generators import (
    GENERATOR_VERSION,
    RANGES,
    TEMPLATES,
    TEMPLATES_BY_ID,
    GenerationReject,
    Problem,
    Sampler,
    Template,
)
from microreason.data.schema import (
    ALL_SPLITS,
    EVAL_SPLITS,
    DataBuildConfig,
    ProblemRecord,
    Split,
    VerifierSpec,
)
from microreason.data.split import (
    ContaminationScanner,
    answer_type,
    equation_fingerprint,
    expression_fingerprint,
    iter_corpus_texts,
    normalize_math,
    normalize_prompt,
)
from microreason.logging_utils import get_logger
from microreason.verify.expr import fmt_value

log = get_logger("microreason.data.build")


class CapacityError(RuntimeError):
    pass


def load_data_config(path: str | Path) -> DataBuildConfig:
    with Path(path).open("r", encoding="utf-8") as f:
        return DataBuildConfig.model_validate(yaml.safe_load(f))


def config_hash(cfg: DataBuildConfig) -> str:
    blob = json.dumps(cfg.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
    return "sha256:" + hashlib.sha256(blob.encode()).hexdigest()


def derive_seed(root_seed: int, template_id: str, split: str, index: int) -> int:
    key = f"{root_seed}|{GENERATOR_VERSION}|{template_id}|{split}|{index}"
    return int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "big")


def number_range_for(split: Split) -> str:
    return "extrapolation" if split == "test_extrapolation" else "in_distribution"


def templates_for(split: Split) -> list[Template]:
    if split == "test_heldout_template":
        return [t for t in TEMPLATES if t.heldout]
    return [t for t in TEMPLATES if not t.heldout]


def target_count(cfg: DataBuildConfig, split: Split, t: Template) -> int:
    override = cfg.count_overrides.get(t.template_id, {})
    return override.get(split, getattr(cfg.counts, split))


def fingerprint(p: Problem) -> str:
    if p.kind == "expression":
        return expression_fingerprint(p.tree)
    return equation_fingerprint(p.lhs, p.rhs)


@dataclass
class Candidate:
    template: Template
    split: Split
    seed: int
    problem: Problem
    fingerprint: str
    math_key: str


@dataclass
class BuildState:
    seen_fp: set[str] = field(default_factory=set)
    seen_prompt: set[str] = field(default_factory=set)
    rejections: Counter = field(default_factory=Counter)


def _generate(
    cfg: DataBuildConfig, split: Split, t: Template, n: int, state: BuildState
) -> list[Candidate]:
    out: list[Candidate] = []
    limit = n * cfg.max_attempts_factor + 1000
    nr = number_range_for(split)
    for idx in range(limit):
        seed = derive_seed(cfg.root_seed, t.template_id, split, idx)
        try:
            p = t.fn(Sampler(random.Random(seed), nr))
        except GenerationReject as e:
            state.rejections[f"sampler:{e}"] += 1
            continue
        fp = fingerprint(p)
        npr = normalize_prompt(p.prompt)
        if fp in state.seen_fp or npr in state.seen_prompt:
            state.rejections["duplicate"] += 1
            continue
        answer = fmt_value(p.answer)
        reason = independent_check(p.kind, p.math_text, p.prompt, answer) or step_check(
            p.kind, p.steps, answer
        )
        if reason:
            state.rejections[reason] += 1
            continue
        state.seen_fp.add(fp)
        state.seen_prompt.add(npr)
        out.append(Candidate(t, split, seed, p, fp, normalize_math(p.math_text)))
        if len(out) == n:
            return out
    raise CapacityError(
        f"{t.template_id}/{split}: only {len(out)} of {n} unique valid problems after "
        f"{limit} attempts; lower the count or widen the ranges"
    )


def _to_record(c: Candidate, index: int) -> ProblemRecord:
    p = c.problem
    major = GENERATOR_VERSION.split(".")[0]
    return ProblemRecord(
        id=f"{c.template.template_id}_v{major}_{c.split}_{index:06d}",
        family=c.template.family,
        template_id=c.template.template_id,
        difficulty=c.template.difficulty,
        prompt=p.prompt,
        solution_steps=p.steps,
        answer_type=answer_type(p.answer),
        final_answer=fmt_value(p.answer),
        verifier_spec=VerifierSpec(
            kind=p.kind, text=p.math_text, variable="x" if p.kind == "linear_equation" else None
        ),
        generator_version=GENERATOR_VERSION,
        seed=c.seed,
        split=c.split,
        number_range=number_range_for(c.split),
        params=p.params,
        semantic_fingerprint=c.fingerprint,
    )


@dataclass
class BuildResult:
    records: dict[Split, list[ProblemRecord]]
    rejections: Counter
    contamination: dict


def build_dataset(cfg: DataBuildConfig) -> BuildResult:
    state = BuildState()
    candidates: dict[tuple[Split, str], list[Candidate]] = {}

    for split in EVAL_SPLITS:
        for t in templates_for(split):
            n = math.ceil(target_count(cfg, split, t) * cfg.overgenerate)
            candidates[(split, t.template_id)] = _generate(cfg, split, t, n, state)
            log.info("candidates %-22s %-28s %d", split, t.template_id, n)

    contaminated: set[str] = set()
    scan_info: dict = {"enabled": cfg.corpus_scan.enabled}
    if cfg.corpus_scan.enabled:
        keys = {c.math_key for cs in candidates.values() for c in cs}
        scanner = ContaminationScanner(keys)
        files: set[str] = set()
        for file_sha, text in iter_corpus_texts(
            cfg.corpus_scan.sources_config, cfg.corpus_scan.raw_dir
        ):
            files.add(file_sha)
            scanner.scan(text)
        contaminated = scanner.found
        scan_info.update(
            corpus_files=sorted(files),
            documents_scanned=scanner.docs_scanned,
            candidate_keys=scanner.n_keys,
            contaminated_keys=len(contaminated),
        )
        log.info("corpus scan: %d docs, %d/%d candidate keys found in corpus",
                 scanner.docs_scanned, len(contaminated), scanner.n_keys)  # fmt: skip

    records: dict[Split, list[ProblemRecord]] = {s: [] for s in ALL_SPLITS}
    dropped: Counter = Counter()
    for (split, tid), cs in candidates.items():
        target = target_count(cfg, split, TEMPLATES_BY_ID[tid])
        clean = [c for c in cs if c.math_key not in contaminated]
        dropped[split] += len(cs) - len(clean)
        if len(clean) < target:
            raise CapacityError(
                f"{tid}/{split}: {len(clean)} clean candidates for target {target}; "
                "raise overgenerate"
            )
        records[split].extend(_to_record(c, i) for i, c in enumerate(clean[:target]))
    scan_info["contaminated_candidates_dropped"] = dict(dropped)

    for t in templates_for("train"):
        cs = _generate(cfg, "train", t, target_count(cfg, "train", t), state)
        records["train"].extend(_to_record(c, i) for i, c in enumerate(cs))
        log.info("generated train %-28s %d", t.template_id, len(cs))

    return BuildResult(records, state.rejections, scan_info)


# ----------------------------------------------------------------------------- writing


def _record_line(r: ProblemRecord) -> str:
    return json.dumps(r.model_dump(mode="json"), sort_keys=True, ensure_ascii=False) + "\n"


def split_path(cfg: DataBuildConfig, split: Split) -> Path:
    sub = "train" if split == "train" else "eval"
    return Path(cfg.out_dir) / sub / f"{split}.jsonl"


def manifest_path(manifest_dir: str | Path, version: str, split: str) -> Path:
    return Path(manifest_dir) / f"math_{split}_{version}.json"


def _code_revision() -> dict:
    try:
        rev = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                             check=True).stdout.strip()  # fmt: skip
        dirty = bool(subprocess.run(["git", "status", "--porcelain"], capture_output=True,
                                    text=True, check=True).stdout.strip())  # fmt: skip
        return {"git_commit": rev, "dirty": dirty}
    except (OSError, subprocess.CalledProcessError):
        return {"git_commit": None, "dirty": None}


def _counts(records: list[ProblemRecord]) -> dict:
    return {
        "by_template": dict(sorted(Counter(r.template_id for r in records).items())),
        "by_family": dict(sorted(Counter(r.family for r in records).items())),
        "by_difficulty": {
            str(k): v for k, v in sorted(Counter(r.difficulty for r in records).items())
        },
        "by_answer_type": dict(sorted(Counter(r.answer_type for r in records).items())),
    }  # fmt: skip


def write_dataset(cfg: DataBuildConfig, result: BuildResult) -> list[Path]:
    written = []
    chash = config_hash(cfg)
    rev = _code_revision()
    for split, recs in result.records.items():
        path = split_path(cfg, split)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = "".join(_record_line(r) for r in recs).encode("utf-8")
        tmp = path.with_suffix(".jsonl.partial")
        tmp.write_bytes(data)
        tmp.replace(path)
        manifest = {
            "schema_version": 1,
            "dataset": cfg.dataset,
            "version": cfg.version,
            "split": split,
            "file": str(path),
            "sha256": hashlib.sha256(data).hexdigest(),
            "record_count": len(recs),
            "generator_version": GENERATOR_VERSION,
            "root_seed": cfg.root_seed,
            "data_config_hash": chash,
            "counts": _counts(recs),
            "corpus_scan": result.contamination,
            "build_rejections": dict(sorted(result.rejections.items())),
            "code_revision": rev,
        }
        mpath = manifest_path(cfg.manifest_dir, cfg.version, split)
        mpath.parent.mkdir(parents=True, exist_ok=True)
        mpath.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        written.append(mpath)
    return written


# ----------------------------------------------------------------------------- audit


def load_split(manifest: dict) -> list[ProblemRecord]:
    data = Path(manifest["file"]).read_bytes()
    if hashlib.sha256(data).hexdigest() != manifest["sha256"]:
        raise ValueError(f"{manifest['file']} does not match manifest sha256")
    return [ProblemRecord.model_validate_json(line) for line in data.decode().splitlines()]


def _param_in_range(role: str, number_range: str, v: int) -> bool:
    lo, hi = RANGES[role][number_range]
    return lo <= abs(v) <= hi


def audit(manifest_dir: str | Path, version: str = "v1") -> dict:
    failures: list[str] = []
    splits: dict[str, list[ProblemRecord]] = {}
    manifests: dict[str, dict] = {}
    for split in ALL_SPLITS:
        mpath = manifest_path(manifest_dir, version, split)
        if not mpath.exists():
            failures.append(f"missing manifest {mpath}")
            continue
        m = json.loads(mpath.read_text())
        manifests[split] = m
        try:
            splits[split] = load_split(m)
        except (OSError, ValueError) as e:
            failures.append(str(e))
            continue
        if len(splits[split]) != m["record_count"]:
            failures.append(f"{split}: record count mismatch")

    report: dict = {"splits": {}, "overlaps": {}, "failures": failures}
    all_ids: Counter = Counter()
    all_seeds: Counter = Counter()
    for split, recs in splits.items():
        bad = Counter()
        for r in recs:
            all_ids[r.id] += 1
            all_seeds[(r.template_id, r.seed)] += 1
            t = TEMPLATES_BY_ID.get(r.template_id)
            if t is None:
                bad["unknown_template"] += 1
                continue
            if r.split != split:
                bad["wrong_split_field"] += 1
            if r.number_range != number_range_for(split):
                bad["wrong_number_range"] += 1
            if t.heldout != (split == "test_heldout_template"):
                bad["heldout_template_leak"] += 1
            for k, v in r.params.items():
                if k not in t.roles or not _param_in_range(t.roles[k], r.number_range, v):
                    bad["param_out_of_range"] += 1
            reason = independent_check(
                r.verifier_spec.kind, r.verifier_spec.text, r.prompt, r.final_answer
            ) or step_check(r.verifier_spec.kind, r.solution_steps, r.final_answer)
            if reason:
                bad[reason] += 1
        for k, v in bad.items():
            failures.append(f"{split}: {v} records with {k}")
        lens = [len(r.prompt) for r in recs] or [0]
        steps = [len(r.solution_steps) for r in recs] or [0]
        report["splits"][split] = {
            "record_count": len(recs),
            "sha256": manifests[split]["sha256"],
            **_counts(recs),
            "prompt_chars": {"mean": round(sum(lens) / len(lens), 1), "max": max(lens)},
            "steps_per_record": {"mean": round(sum(steps) / len(steps), 2), "max": max(steps)},
        }

    dup_ids = sum(1 for v in all_ids.values() if v > 1)
    dup_seeds = sum(1 for v in all_seeds.values() if v > 1)
    if dup_ids:
        failures.append(f"{dup_ids} duplicate ids")
    if dup_seeds:
        failures.append(f"{dup_seeds} duplicate (template, seed) pairs")

    keys = {
        "fingerprint": lambda r: r.semantic_fingerprint,
        "prompt": lambda r: normalize_prompt(r.prompt),
    }
    for a, b in combinations(splits, 2):
        for name, fn in keys.items():
            n = len({fn(r) for r in splits[a]} & {fn(r) for r in splits[b]})
            report["overlaps"][f"{a}~{b}:{name}"] = n
            if n:
                failures.append(f"{n} {name} overlaps between {a} and {b}")
    for split, recs in splits.items():
        n = len(recs) - len({r.semantic_fingerprint for r in recs})
        if n:
            failures.append(f"{split}: {n} duplicate fingerprints within split")

    any_m = next(iter(manifests.values()), {})
    report["corpus_scan"] = any_m.get("corpus_scan")
    report["build_rejections"] = any_m.get("build_rejections")
    report["passed"] = not failures
    return report


# ----------------------------------------------------------------------------- freeze


def freeze(manifest_dir: str | Path, version: str = "v1") -> Path:
    report = audit(manifest_dir, version)
    if not report["passed"]:
        raise ValueError("audit failed; refusing to freeze:\n" + "\n".join(report["failures"]))
    suites = {
        s: {"sha256": report["splits"][s]["sha256"],
            "record_count": report["splits"][s]["record_count"]}
        for s in EVAL_SPLITS
    }  # fmt: skip
    frozen = {"dataset": "math", "version": version, "suites": suites}
    path = Path(manifest_dir) / f"math_eval_frozen_{version}.json"
    if path.exists():
        existing = json.loads(path.read_text())
        if existing["suites"] != suites:
            raise ValueError(f"{path} already freezes different suites; refusing to overwrite")
        return path
    path.write_text(json.dumps(frozen, indent=2, sort_keys=True) + "\n")
    return path


def frozen_suites(manifest_dir: str | Path, version: str = "v1") -> dict | None:
    path = Path(manifest_dir) / f"math_eval_frozen_{version}.json"
    return json.loads(path.read_text())["suites"] if path.exists() else None


def guard_against_frozen(cfg: DataBuildConfig, result: BuildResult) -> None:
    """Refuse to write a build whose evaluation suites differ from frozen ones."""
    frozen = frozen_suites(cfg.manifest_dir, cfg.version)
    if frozen is None:
        return
    for split in EVAL_SPLITS:
        data = "".join(_record_line(r) for r in result.records[split]).encode()
        if hashlib.sha256(data).hexdigest() != frozen[split]["sha256"]:
            raise ValueError(f"build would change frozen suite '{split}'; refusing to write")


def summarize(result: BuildResult) -> dict:
    return {
        "records": {s: len(r) for s, r in result.records.items()},
        "rejections": dict(result.rejections),
        "corpus_scan": result.contamination,
    }
