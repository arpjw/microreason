"""Training-only math stream: frozen-suite exclusion, determinism, capacity, shards."""

import hashlib
import json
from pathlib import Path

import pytest
from test_generators import small_cfg

from microreason.data import build
from microreason.data.split import normalize_prompt
from microreason.data.stream import (
    StreamConfig,
    audit_stream,
    build_stream,
    iter_stream,
    shard_of,
)
from microreason.data.tokenized import TokenizedDataset, tokenize_math_stream
from microreason.tokenizer import MicroTokenizer

TOK_DIR = Path(__file__).resolve().parents[1] / "artifacts" / "tokenizer" / "v1"
TARGETS = {"add_ab": 300, "ax_plus_b_equals_c": 300, "a_paren_x_plus_b_equals_c": 200}


@pytest.fixture
def frozen_eval(tmp_path):
    cfg = small_cfg(tmp_path / "eval")
    build.write_dataset(cfg, build.build_dataset(cfg))
    build.freeze(cfg.manifest_dir)
    return cfg


def stream_cfg(tmp_path, eval_cfg, name="s", **over):
    base = dict(
        root_seed=7,
        shard_a_fraction=0.5,
        targets=TARGETS,
        eval_manifest_dir=eval_cfg.manifest_dir,
        out_path=str(tmp_path / name / "stream.jsonl"),
        manifest_path=str(tmp_path / name / "manifest.json"),
    )
    base.update(over)
    return StreamConfig.model_validate(base)


def eval_keys(eval_cfg):
    fps, prompts = set(), set()
    for split in ("val", "test_id", "test_extrapolation", "test_heldout_template"):
        m = json.loads(build.manifest_path(eval_cfg.manifest_dir, "v1", split).read_text())
        for r in build.load_split(m):
            fps.add(r.semantic_fingerprint)
            prompts.add(normalize_prompt(r.prompt))
    return fps, prompts


def test_stream_excludes_frozen_suites_and_passes_audit(tmp_path, frozen_eval):
    cfg = stream_cfg(tmp_path, frozen_eval)
    m = build_stream(cfg)
    assert m["record_count"] == 800 and m["capacity_limited"] == []
    fps, prompts = eval_keys(frozen_eval)
    recs = list(iter_stream(m))
    assert not {r.semantic_fingerprint for r in recs} & fps
    assert not {normalize_prompt(r.prompt) for r in recs} & prompts
    assert all(r.split == "train_stream" for r in recs)
    report = audit_stream(cfg, sample_fraction=1.0)
    assert report["passed"], report
    assert report["rechecked_sample"] == 800


def test_stream_is_deterministic(tmp_path, frozen_eval):
    a = build_stream(stream_cfg(tmp_path, frozen_eval, "a"))
    b = build_stream(stream_cfg(tmp_path, frozen_eval, "b"))
    assert a["sha256"] == b["sha256"]


def test_capacity_limited_template_returns_partial(tmp_path, frozen_eval):
    cfg = stream_cfg(tmp_path, frozen_eval, targets={"add_ab": 50_000}, max_attempts_factor=1)
    m = build_stream(cfg)
    assert m["capacity_limited"] == ["add_ab"]
    assert 0 < m["achieved"]["add_ab"] < 50_000
    assert audit_stream(cfg, sample_fraction=0.05)["passed"]


def test_refuses_unfrozen_suites(tmp_path):
    cfg = small_cfg(tmp_path / "eval")
    build.write_dataset(cfg, build.build_dataset(cfg))
    with pytest.raises(ValueError, match="not frozen"):
        build_stream(stream_cfg(tmp_path, cfg))


def test_refuses_heldout_templates(tmp_path, frozen_eval):
    cfg = stream_cfg(tmp_path, frozen_eval, targets={"heldout_nested_product": 10})
    with pytest.raises(ValueError, match="held-out"):
        build_stream(cfg)


def test_refuses_tampered_eval_suite(tmp_path, frozen_eval):
    p = build.split_path(frozen_eval, "val")
    p.write_text(p.read_text().replace('"final_answer": "', '"final_answer": "1', 1))
    with pytest.raises(ValueError):
        build_stream(stream_cfg(tmp_path, frozen_eval))


def test_audit_catches_injected_eval_problem(tmp_path, frozen_eval):
    cfg = stream_cfg(tmp_path, frozen_eval)
    build_stream(cfg)
    val = build.split_path(frozen_eval, "val").read_text().splitlines()[0]
    leaked = json.loads(val)
    leaked.update(split="train_stream", id="leak")
    out = Path(cfg.out_path)
    data = out.read_bytes() + (json.dumps(leaked, sort_keys=True) + "\n").encode()
    out.write_bytes(data)
    m = json.loads(Path(cfg.manifest_path).read_text())
    m.update(sha256=hashlib.sha256(data).hexdigest(), record_count=m["record_count"] + 1)
    Path(cfg.manifest_path).write_text(json.dumps(m))
    report = audit_stream(cfg, sample_fraction=0.0)
    assert not report["passed"]
    assert report["failures"]["fingerprint_in_eval"] == 1


def test_shards_are_disjoint_and_complete(tmp_path, frozen_eval):
    cfg = stream_cfg(tmp_path, frozen_eval)
    m = build_stream(cfg)
    ids = [r.id for r in iter_stream(m)]
    a = {i for i in ids if shard_of(i, 0.5) == "a"}
    assert 0 < len(a) < len(ids)
    tok = MicroTokenizer.load(TOK_DIR)
    da = TokenizedDataset(tokenize_math_stream(tok, cfg.manifest_path, "a", tmp_path / "ta")
                          and tmp_path / "ta")  # fmt: skip
    db = TokenizedDataset(tokenize_math_stream(tok, cfg.manifest_path, "b", tmp_path / "tb")
                          and tmp_path / "tb")  # fmt: skip
    assert len(da) == len(a) and len(da) + len(db) == len(ids)
    assert da.manifest["split"] == db.manifest["split"] == "train"
    assert da.manifest["tokenizer_sha256"] == tok.sha256
