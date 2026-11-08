import hashlib
import json
import random

import pytest

from microreason.data import build
from microreason.data.checks import independent_check, step_check
from microreason.data.generators import RANGES, TEMPLATES, Sampler
from microreason.data.schema import ALL_SPLITS, DataBuildConfig
from microreason.data.split import ContaminationScanner, equation_fingerprint, normalize_math
from microreason.verify.exact import verify_completion
from microreason.verify.expr import fmt_value, parse_equation


def small_cfg(tmp_path, **over):
    base = dict(
        root_seed=7,
        counts=dict(train=40, val=8, test_id=8, test_extrapolation=8, test_heldout_template=10),
        corpus_scan={"enabled": False},
        out_dir=str(tmp_path / "gen"),
        manifest_dir=str(tmp_path / "manifests"),
    )
    base.update(over)
    return DataBuildConfig.model_validate(base)


@pytest.mark.parametrize("t", TEMPLATES, ids=lambda t: t.template_id)
@pytest.mark.parametrize("nr", ["in_distribution", "extrapolation"])
def test_every_template_passes_independent_and_step_checks(t, nr):
    for seed in range(150):
        try:
            p = t.fn(Sampler(random.Random(seed), nr))
        except build.GenerationReject:
            continue
        a = fmt_value(p.answer)
        assert independent_check(p.kind, p.math_text, p.prompt, a) is None, p.prompt
        assert step_check(p.kind, p.steps, a) is None, (p.prompt, p.steps)
        for k, v in p.params.items():
            lo, hi = RANGES[t.roles[k]][nr]
            assert lo <= abs(v) <= hi, (t.template_id, k, v)


@pytest.mark.parametrize("t", TEMPLATES, ids=lambda t: t.template_id)
def test_gold_completion_scores_correct(t):
    p = t.fn(Sampler(random.Random(1), "in_distribution"))
    spec = {"kind": p.kind, "text": p.math_text}
    steps = "\n".join(p.steps)
    completion = f"[steps]\n{steps}\n[/steps]\n[answer]{fmt_value(p.answer)}[/answer][eos]"
    assert verify_completion(fmt_value(p.answer), spec, completion).status == "correct"


def test_build_is_byte_identical(tmp_path):
    hashes = []
    for run in ("a", "b"):
        cfg = small_cfg(tmp_path / run)
        build.write_dataset(cfg, build.build_dataset(cfg))
        hashes.append(
            {
                s: hashlib.sha256(build.split_path(cfg, s).read_bytes()).hexdigest()
                for s in ALL_SPLITS
            }
        )
    assert hashes[0] == hashes[1]


def test_eval_suites_do_not_depend_on_train_size(tmp_path):
    a = build.build_dataset(small_cfg(tmp_path))
    counts = dict(train=80, val=8, test_id=8, test_extrapolation=8, test_heldout_template=10)
    b = build.build_dataset(small_cfg(tmp_path, counts=counts))
    for s in ("test_heldout_template", "test_extrapolation", "test_id", "val"):
        assert a.records[s] == b.records[s]


def test_different_seed_changes_data(tmp_path):
    a = build.build_dataset(small_cfg(tmp_path))
    b = build.build_dataset(small_cfg(tmp_path, root_seed=8))
    assert a.records["test_id"] != b.records["test_id"]


def test_audit_passes_and_catches_tampering(tmp_path):
    cfg = small_cfg(tmp_path)
    build.write_dataset(cfg, build.build_dataset(cfg))
    report = build.audit(cfg.manifest_dir)
    assert report["passed"], report["failures"]
    assert all(v == 0 for v in report["overlaps"].values())
    assert set(report["splits"]["test_heldout_template"]["by_template"]) == {
        t.template_id for t in TEMPLATES if t.heldout
    }

    # Copy one test record into train: the audit must flag the leak.
    test_path = build.split_path(cfg, "test_id")
    train_path = build.split_path(cfg, "train")
    leaked = json.loads(test_path.read_text().splitlines()[0])
    leaked["split"] = "train"
    leaked["id"] = "leak"
    data = train_path.read_bytes() + (json.dumps(leaked, sort_keys=True) + "\n").encode()
    train_path.write_bytes(data)
    mpath = build.manifest_path(cfg.manifest_dir, "v1", "train")
    m = json.loads(mpath.read_text())
    m["sha256"] = hashlib.sha256(data).hexdigest()
    m["record_count"] += 1
    mpath.write_text(json.dumps(m))
    report = build.audit(cfg.manifest_dir)
    assert not report["passed"]
    assert any("fingerprint overlaps between test_id and train" in f for f in report["failures"])


def test_audit_detects_modified_file(tmp_path):
    cfg = small_cfg(tmp_path)
    build.write_dataset(cfg, build.build_dataset(cfg))
    p = build.split_path(cfg, "val")
    p.write_text(p.read_text().replace('"final_answer": "', '"final_answer": "1', 1))
    assert not build.audit(cfg.manifest_dir)["passed"]


def test_freeze_then_refuse_changed_suites(tmp_path):
    cfg = small_cfg(tmp_path)
    build.write_dataset(cfg, build.build_dataset(cfg))
    path = build.freeze(cfg.manifest_dir)
    assert path.exists()
    assert build.freeze(cfg.manifest_dir) == path  # idempotent

    other = small_cfg(tmp_path, root_seed=99)
    with pytest.raises(ValueError, match="frozen suite"):
        build.guard_against_frozen(other, build.build_dataset(other))


def test_capacity_error_when_space_too_small(tmp_path):
    counts = dict(train=40, val=8, test_id=8, test_extrapolation=8, test_heldout_template=10)
    cfg = small_cfg(tmp_path, counts=counts, count_overrides={"add_ab": {"test_id": 10**6}},
                    max_attempts_factor=1)  # fmt: skip
    with pytest.raises(build.CapacityError):
        build.build_dataset(cfg)


def test_equation_fingerprint_invariances():
    def fp(t):
        return equation_fingerprint(*parse_equation(t))

    assert fp("4x + 7 = 31") == fp("31 = 4x + 7") == fp("7 + 4x = 31")
    assert fp("3(x + 4) = 27") == fp("3x + 12 = 27")  # trivial expansion
    # Same solution, different exercise: must stay distinct.
    assert fp("x + 3 = 5") != fp("x + 4 = 6")
    assert fp("5x = 35") != fp("7x = 49")
    assert fp("8x + 14 = 62") != fp("4x + 7 = 31")


def test_contamination_scanner_boundaries():
    keys = [normalize_math(k) for k in ("3 + 5", "4x + 7 = 31", "12 * (-3)")]
    s = ContaminationScanner(keys)
    s.scan("We know 3+5 = 8.")  # hit
    s.scan("Solve 24x + 7 = 31 now")  # glued to a digit: no hit
    s.scan("13 + 52 is big")  # no hit
    s.scan("compute 12 × (−3)")  # unicode operators normalize: hit
    assert s.found == {"3+5", "12*(-3)"}
