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


# ----------------------------------------------------------------------------- Phase C RL pool

FAMILY_P = {"expression_1op": 0.947, "expression_2op": 0.796, "expression_3op": 0.637,
            "expression_4op": 0.505, "expression_5op": 0.366, "linear_equation": 0.958}  # fmt: skip


def _pool_cfg(tmp_path, workers=2, records=1500):
    cfg = _cfg(tmp_path)
    pool = b3.RLPoolV3(records=records, family_accuracy=FAMILY_P, part_size=1000,
                       workers=workers, out_path=str(tmp_path / "math_v3/train_rl/train_rl.jsonl"),
                       manifest_path=str(tmp_path / "manifests/math_train_rl_v3.json"))  # fmt: skip
    return cfg.model_copy(update={"rl_pool": pool})


def test_rl_pool_is_weighted_partitioned_and_eval_free(tmp_path):
    from collections import Counter

    from microreason.rl.pool import load_partition, partition_of

    hashes = []
    for w in (1, 2):
        cfg = _pool_cfg(tmp_path / f"w{w}", workers=w)
        b3.write_v3(cfg, b3.build_v3(cfg))
        b3.freeze_v3(cfg.manifest_dir)
        m = b3.build_rl_pool_v3(cfg)
        hashes.append(m["sha256"])
    assert hashes[0] == hashes[1]
    assert m["record_count"] == 1500 and m["purpose"] == "rl_pool"
    keep = m["family_keep_probability"]
    assert keep["expression_4op"] == 1.0 and keep["linear_equation"] < 0.35
    recs = [json.loads(x) for x in open(m["file"])]
    assert all(r["id"].startswith("v3_train_rl_") for r in recs)
    fam = Counter(r["family"] for r in recs)
    # equations are ~36% of unweighted problems; the weights cut them to well under that
    assert fam["linear_equation"] / len(recs) < 0.2
    assert fam["expression_4op"] > fam["expression_1op"]
    eval_fps = set()
    for split in b3.V3_EVAL_SPLITS:
        em = json.loads(b3.manifest_path(cfg.manifest_dir, "v3", split).read_text())
        eval_fps |= {r.semantic_fingerprint for r in b3.load_split_v3(em)}
    assert not {r["semantic_fingerprint"] for r in recs} & eval_fps
    grpo, _ = load_partition(cfg.rl_pool.manifest_path, "grpo")
    assert grpo and all(partition_of(r["id"]) == "grpo" for r in grpo)
    assert set(m["partitions"]) >= {"rule", "gate", "grpo", "sft_extra"}


def test_rl_pool_requires_declared_accuracies(tmp_path):
    cfg = _cfg(tmp_path)
    with pytest.raises(ValueError, match="family_accuracy"):
        b3.build_rl_pool_v3(cfg)


def test_grpo_and_extra_sft_run_on_the_v3_pool(tmp_path):
    import torch

    from microreason.checkpoint import load_checkpoint, save_checkpoint
    from microreason.config import ResolvedConfig, StageConfig, load_model_config
    from microreason.model import build_model
    from microreason.rl.config import GRPOConfig, ResolvedRL
    from microreason.rl.grpo import GRPOTrainer
    from microreason.rl.pool import load_partition
    from microreason.tokenizer import MicroTokenizer
    from microreason.train.sft import build_sft_trainer

    tok_dir = "artifacts/tokenizer/v2"
    tok = MicroTokenizer.load(tok_dir)
    cfg = _pool_cfg(tmp_path, records=1500)
    b3.write_v3(cfg, b3.build_v3(cfg))
    b3.freeze_v3(cfg.manifest_dir)
    b3.build_rl_pool_v3(cfg)
    smoke = "configs/model_smoke.yaml"
    mc = load_model_config(smoke)
    torch.manual_seed(0)
    m = build_model(mc)
    prov = {"config_hash": "p", "tokenizer_sha256": tok.sha256}
    opt = torch.optim.AdamW(m.parameters())
    parent = save_checkpoint(tmp_path / "parent", 1, m, opt, {"step": 1}, prov)
    pool, _ = load_partition(cfg.rl_pool.manifest_path, "grpo")
    g = GRPOConfig(model=smoke, tokenizer=tok_dir, parent=str(parent), precision="fp32",
                   sampling={"group_size": 2, "max_new_tokens": 8, "generation_batch_size": 8,
                             "kv_cache": True},
                   prompts_per_step=2, steps=2, warmup_steps=0, checkpoint_every_steps=10,
                   log_every_steps=1, dynamic_sampling=True, sample_chunk_prompts=4,
                   max_sample_rounds=2)  # fmt: skip
    res = GRPOTrainer(ResolvedRL(stage=g, model=mc), tmp_path / "grpo", "cpu", pool=pool).train()
    assert res["status"] == "completed"
    stage = StageConfig.model_validate(dict(
        stage="extra_sft", model=smoke, tokenizer=tok_dir, parent=str(parent),
        sft={"train_manifest": cfg.rl_pool.manifest_path, "stream_partition": "sft_extra",
             "val_manifest": str(b3.manifest_path(cfg.manifest_dir, "v3", "val")),
             "num_examples": 200, "val_examples_per_format": 4},
        seed=0, precision="fp32", sequence_length=384, effective_batch_tokens=384 * 4,
        micro_batch_size=4, max_examples=10_000_000, max_wall_seconds=1.0, log_every_steps=1,
        eval_every_steps=10_000, checkpoint_every_steps=10_000,
        optimizer={"peak_lr": 1e-4}, scheduler={"name": "constant", "warmup_fraction": 0.0},
    ))  # fmt: skip
    out = build_sft_trainer(ResolvedConfig(stage=stage, model=mc), tmp_path / "xsft", "cpu").train()
    assert out.final_checkpoint is not None
    _, _, cm = load_checkpoint(out.final_checkpoint)
    assert cm["tokenizer_sha256"] == tok.sha256
