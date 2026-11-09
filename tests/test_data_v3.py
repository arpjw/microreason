"""Data v3 (protocol v3): tree generator, shape split, build/audit/freeze, stream, tokenizer v2."""

import json
import random

import pytest

from microreason.data import build_v3 as b3
from microreason.data.checks import independent_check, step_check
from microreason.data.generators import GenerationReject
from microreason.data.trees import (
    DIGIT_RANGES,
    all_eq_shapes,
    all_expr_shapes,
    coverage_report,
    expr_shape,
    is_eligible,
    sample_problem,
    shape_set,
)
from microreason.verify.expr import fmt_value


def _samples(n, nr="in_distribution", start=0):
    out = []
    for i in range(start, start + 50 * n):
        try:
            out.append(sample_problem(random.Random(i), nr))
        except GenerationReject:
            continue
        if len(out) == n:
            break
    return out


# ----------------------------------------------------------------------------- generator


def test_sampling_is_deterministic():
    a = [(p.prompt, p.steps, p.answer, s) for p, s in _samples(50)]
    b = [(p.prompt, p.steps, p.answer, s) for p, s in _samples(50)]
    assert a == b


@pytest.mark.parametrize("nr", ["in_distribution", "extrapolation"])
def test_every_sample_verifies_and_traces_check(nr):
    kinds = set()
    for p, _ in _samples(400, nr):
        a = fmt_value(p.answer)
        assert independent_check(p.kind, p.math_text, p.prompt, a) is None, p.prompt
        assert step_check(p.kind, p.steps, a) is None, (p.prompt, p.steps)
        kinds.add(p.kind)
    assert kinds == {"expression", "linear_equation"}


def test_number_ranges_are_disjoint_by_digit_count():
    for nr, digits in (("in_distribution", {1, 2, 3}), ("extrapolation", {4, 5})):
        for p, _ in _samples(200, nr):
            if p.kind != "expression":
                continue
            for v in p.params.values():
                assert len(str(abs(v))) in digits, (nr, v)
    lo_ext = DIGIT_RANGES["extrapolation"][4][0]
    assert lo_ext > max(hi for _, hi in DIGIT_RANGES["in_distribution"].values())


def test_equation_traces_use_the_declared_step_vocabulary():
    verbs = set()
    for p, _ in _samples(600):
        if p.kind == "linear_equation":
            verbs |= {s.split(":")[0].split()[0] for s in p.steps}
    assert verbs <= {"Expand", "Combine", "Subtract", "Add", "Divide"}
    assert {"Expand", "Combine", "Divide"} <= verbs


def test_expression_shape_matches_the_tree():
    for p, s in _samples(100):
        if p.kind == "expression":
            assert s == expr_shape(p.tree) and s.count("n") == len(p.params)


# ----------------------------------------------------------------------------- shape split


def test_shape_split_is_fixed_and_covered():
    cov = coverage_report()
    assert cov["passed"], cov["failures"]
    # Pinned: changing the split rule or grammar must be a deliberate, versioned change.
    assert {k: cov["expression"][k] for k in ("train", "heldout_val", "heldout_test")} == {
        "train": 37768, "heldout_val": 4591, "heldout_test": 4589}  # fmt: skip
    assert {k: cov["equation"][k] for k in ("train", "heldout_val", "heldout_test")} == {
        "train": 137, "heldout_val": 10, "heldout_test": 9}  # fmt: skip


def test_only_eligible_shapes_are_held_out_and_sets_are_disjoint():
    for s in all_expr_shapes() + all_eq_shapes():
        if not is_eligible(s):
            assert shape_set(s) == "train"
    small = [s for s in all_expr_shapes() if s.count("n") <= 3]
    assert all(shape_set(s) == "train" for s in small)


def test_split_constraints_hold_for_generated_candidates():
    for split, want in b3.REQUIRED_SET.items():
        n = 0
        for i in range(4000):
            c, _ = b3.generate_one(11, split, i)
            if c is None:
                continue
            assert shape_set(c.shape) == want, (split, c.shape)
            n += 1
            if n == 20:
                break
        assert n == 20, split


# ----------------------------------------------------------------------------- build / audit


def _cfg(tmp_path, workers=2):
    return b3.DataV3Config.model_validate({
        "root_seed": 7,
        "counts": {"train": 120, "val": 60, "test_id": 60, "test_extrapolation": 60,
                   "val_heldout_shape": 40, "test_heldout_shape": 40},
        "corpus_scan": {"enabled": False},
        "out_dir": str(tmp_path / "math_v3"), "manifest_dir": str(tmp_path / "manifests"),
        "stream": {"records": 600, "part_size": 500, "workers": workers,
                   "out_path": str(tmp_path / "math_v3/train_stream/train_stream.jsonl"),
                   "manifest_path": str(tmp_path / "manifests/math_train_stream_v3.json")},
    })  # fmt: skip


def test_build_audit_freeze_and_tamper_detection(tmp_path):
    cfg = _cfg(tmp_path)
    b3.write_v3(cfg, b3.build_v3(cfg))
    rep = b3.audit_v3(cfg.manifest_dir)
    assert rep["passed"], rep["failures"]
    frozen = b3.freeze_v3(cfg.manifest_dir)
    assert b3.freeze_v3(cfg.manifest_dir) == frozen  # idempotent
    suites = json.loads(frozen.read_text())["suites"]
    assert set(suites) == {"val", "test_id", "test_extrapolation", "val_heldout_shape",
                           "test_heldout_shape"}  # fmt: skip
    # all held-out-test shapes are distinct from every train-visible split's shapes
    shapes = {}
    for split in ("train", "val", "test_id", "test_heldout_shape", "val_heldout_shape"):
        m = json.loads(b3.manifest_path(cfg.manifest_dir, "v3", split).read_text())
        shapes[split] = {r.template_id for r in b3.load_split_v3(m)}
    assert not shapes["test_heldout_shape"] & (shapes["train"] | shapes["val"])
    assert not shapes["test_heldout_shape"] & shapes["val_heldout_shape"]
    # tampering with a record breaks the manifest hash
    f = tmp_path / "math_v3/eval/val.jsonl"
    f.write_text(f.read_text().replace('"difficulty": 1', '"difficulty": 2', 1))
    with pytest.raises(ValueError, match="hash"):
        b3.audit_v3(cfg.manifest_dir)


def test_stream_excludes_eval_and_is_worker_count_invariant(tmp_path):
    hashes = []
    for w in (1, 2):
        d = tmp_path / f"w{w}"
        cfg = _cfg(d, workers=w)
        b3.write_v3(cfg, b3.build_v3(cfg))
        b3.freeze_v3(cfg.manifest_dir)
        m = b3.build_stream_v3(cfg)
        assert m["record_count"] == 600 and set(m["shards"]) == {"rule", "a", "b"}
        hashes.append(m["sha256"])
        eval_fps = set()
        for split in b3.V3_EVAL_SPLITS:
            em = json.loads(b3.manifest_path(cfg.manifest_dir, "v3", split).read_text())
            eval_fps |= {r.semantic_fingerprint for r in b3.load_split_v3(em)}
        recs = [json.loads(x) for x in open(m["file"])]
        assert not {r["semantic_fingerprint"] for r in recs} & eval_fps
        assert all(shape_set(r["template_id"]) == "train" for r in recs)
        assert len({r["id"] for r in recs}) == 600
    assert hashes[0] == hashes[1]


def test_unfrozen_suites_block_the_stream(tmp_path):
    cfg = _cfg(tmp_path)
    b3.write_v3(cfg, b3.build_v3(cfg))
    with pytest.raises(ValueError, match="not frozen"):
        b3.build_stream_v3(cfg)


# ----------------------------------------------------------------------------- tokenizer v2


def test_tokenizer_v2_gives_one_token_per_digit(tmp_path):
    from microreason.tokenizer import (
        MicroTokenizer,
        TokenizerConfig,
        save_tokenizer,
        train_tokenizer,
    )

    texts = [p.prompt + " " + " ".join(p.steps) for p, _ in _samples(300)]
    texts += ["The year 1999 had 365 days and 12 months."] * 20
    cfg = TokenizerConfig(version="t", vocab_size=400, digits="individual",
                          out_dir=str(tmp_path / "tok"))  # fmt: skip
    save_tokenizer(train_tokenizer(cfg, texts), cfg, {})
    tok = MicroTokenizer.load(tmp_path / "tok")
    digit_ids = {d: tok.encode(d) for d in "0123456789"}
    assert all(len(v) == 1 for v in digit_ids.values())
    for text, number in (("What is -330 + 167?", "167"), ("x = 12/5", "12"),
                         ("Compute: 98765 * 3", "98765")):  # fmt: skip
        ids = tok.encode(text)
        assert tok.decode(ids) == text
        seq = [digit_ids[d][0] for d in number]
        assert any(ids[i : i + len(seq)] == seq for i in range(len(ids))), text
    manifest = json.loads((tmp_path / "tok/tokenizer_manifest.json").read_text())
    assert manifest["pre_tokenization"].startswith("Digits(individual_digits=True)")


# ----------------------------------------------------------------------------- evaluation


def test_eval_on_v3_suites_with_cached_generator_and_sealing(tmp_path):
    import torch

    from microreason.checkpoint import save_checkpoint
    from microreason.config import load_model_config
    from microreason.eval.report import SealedSuiteError, run_eval
    from microreason.model import build_model
    from microreason.tokenizer import MicroTokenizer

    cfg = _cfg(tmp_path)
    b3.write_v3(cfg, b3.build_v3(cfg))
    b3.freeze_v3(cfg.manifest_dir)
    tok = MicroTokenizer.load("artifacts/tokenizer/v1")
    torch.manual_seed(0)
    model = build_model(load_model_config("configs/model_smoke.yaml"))
    prov = {"config_hash": "x", "tokenizer_sha256": tok.sha256}
    opt = torch.optim.AdamW(model.parameters())
    ckpt = save_checkpoint(tmp_path / "run", 1, model, opt, {"step": 1}, prov)
    common = dict(manifest_dir=cfg.manifest_dir, data_version="v3", max_new_tokens=8,
                  batch_size=16, limit=12)  # fmt: skip
    s = run_eval("val_heldout_shape", ckpt, "sft-show-work-v1", tmp_path / "e1", kv_cache=True,
                 samples=2, **common)  # fmt: skip
    assert s["data_version"] == "v3" and s["protocol"] == "eval-v2-cached"
    assert s["decode"]["kv_cache"] is True and s["greedy"]["prompts"] == 12
    run_eval("val_heldout_shape", ckpt, "sft-show-work-v1", tmp_path / "e2", **common)
    rows = lambda d: [json.loads(x)["raw_completion"] for x in open(d / "rows.jsonl")]  # noqa: E731
    assert rows(tmp_path / "e1")[:12] == rows(tmp_path / "e2")  # cached == uncached greedy
    with pytest.raises(SealedSuiteError):
        run_eval("test_heldout_shape", ckpt, "sft-show-work-v1", tmp_path / "e3", **common)
    with pytest.raises(ValueError, match="not an evaluation suite of data-v3"):
        run_eval("test_heldout_template", ckpt, "sft-show-work-v1", tmp_path / "e4",
                 allow_test=True, **common)  # fmt: skip
