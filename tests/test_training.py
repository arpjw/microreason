"""Stage 0 gates for the common training loop (spec §6): overfit, accumulation, resume."""

import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch

from microreason.checkpoint import (
    CheckpointError,
    checkpoint_dir,
    latest_checkpoint,
    load_checkpoint,
    verify_checkpoint,
)
from microreason.config import (
    ModelConfig,
    ResolvedConfig,
    SchedulerConfig,
    StageConfig,
    load_model_config,
)
from microreason.data.tokenized import TokenizedDataset, write_tokenized
from microreason.train.common import NonFiniteError, Trainer, lr_at, param_groups
from microreason.train.data import PackedStream, Source, fixed_eval_batches

CONFIGS = Path(__file__).resolve().parents[1] / "configs"
BOS, EOS = 1, 2


@pytest.fixture(scope="module")
def smoke_model_cfg() -> ModelConfig:
    return load_model_config(CONFIGS / "model_smoke.yaml")


def make_dataset(path, docs, split="train", name=None):
    write_tokenized(path, iter(docs), {"name": name or Path(path).name, "split": split})
    return TokenizedDataset(path, verify=True)


def random_docs(seed, n, lo=5, hi=60):
    rng = np.random.default_rng(seed)
    return [rng.integers(9, 8192, rng.integers(lo, hi)).tolist() for _ in range(n)]


def resolved(model_cfg, **stage):
    base = dict(
        stage="smoke",
        model="configs/model_smoke.yaml",
        seed=0,
        precision="fp32",
        sequence_length=32,
        effective_batch_tokens=32 * 8,
        micro_batch_size=4,
        max_steps=6,
        log_every_steps=1,
        eval_every_steps=3,
        checkpoint_every_steps=3,
        optimizer={"peak_lr": 1e-3},
        scheduler={"warmup_fraction": 0.2},
    )
    base.update(stage)
    return ResolvedConfig(stage=StageConfig.model_validate(base), model=model_cfg)


@pytest.fixture
def data(tmp_path):
    a = make_dataset(tmp_path / "text", random_docs(1, 300), name="text")
    b = make_dataset(tmp_path / "math", random_docs(2, 500, 5, 20), name="math")
    v = make_dataset(tmp_path / "val", random_docs(3, 50), split="val", name="val")
    return a, b, v


def make_trainer(cfg, run_dir, data, model=None):
    a, b, v = data
    stream = PackedStream(
        [Source("text", a, 0.7), Source("math", b, 0.3)],
        cfg.stage.sequence_length,
        cfg.stage.seed,
        BOS,
        EOS,
    )
    micro = cfg.stage.micro_batch_size or cfg.stage.sequences_per_step
    evals = {"val": fixed_eval_batches(v, 8, cfg.stage.sequence_length, micro, BOS, EOS)}
    return Trainer(cfg, run_dir, stream, evals, "cpu", {"test": True}, model)


# ----------------------------------------------------------------------------- schedule


def test_lr_schedule_shape():
    s = SchedulerConfig(name="cosine", warmup_fraction=0.1, min_lr_fraction=0.1)
    lrs = [lr_at(i, 100, 1.0, s) for i in range(100)]
    assert lrs[0] == pytest.approx(0.1)
    assert lrs[9] == pytest.approx(1.0)
    assert max(lrs) == pytest.approx(1.0)
    assert lrs[-1] == pytest.approx(0.1)
    assert all(a >= b - 1e-12 for a, b in zip(lrs[9:], lrs[10:], strict=False))


def test_weight_decay_policy(smoke_model_cfg):
    from microreason.model import build_model

    m = build_model(smoke_model_cfg)
    decay, no_decay = param_groups(m, 0.1)
    assert no_decay["weight_decay"] == 0.0
    assert any(p is m.embed.weight for p in no_decay["params"])
    assert all(p.ndim == 2 for p in decay["params"])
    assert all(p.ndim == 1 or p is m.embed.weight for p in no_decay["params"])


# ----------------------------------------------------------------------------- stream


def test_stream_deterministic_and_mixture(data):
    a, b, _ = data

    def stream():
        return PackedStream([Source("text", a, 0.7), Source("math", b, 0.3)], 32, 5, BOS, EOS)

    s1, s2 = stream(), stream()
    x1 = torch.cat([s1.next_batch(8) for _ in range(40)])
    x2 = torch.cat([s2.next_batch(8) for _ in range(40)])
    assert torch.equal(x1, x2)
    mix = s1.mixture_report()
    assert abs(mix["text"]["actual"] - 0.7) < 0.02
    assert abs(mix["math"]["actual"] - 0.3) < 0.02


def test_stream_state_roundtrip_mid_buffer(data):
    a, b, _ = data
    srcs = [Source("text", a, 0.7), Source("math", b, 0.3)]
    s = PackedStream(srcs, 32, 5, BOS, EOS)
    for _ in range(13):
        s.next_batch(3)
    state = json.loads(json.dumps(s.state_dict()))  # must be plain data
    assert state["buffer"]  # resuming mid-document
    want = torch.cat([s.next_batch(3) for _ in range(20)])
    r = PackedStream(srcs, 32, 5, BOS, EOS)
    r.load_state_dict(state)
    assert torch.equal(torch.cat([r.next_batch(3) for _ in range(20)]), want)
    assert r.digest == s.digest


def test_stream_visits_every_doc_once_per_epoch(tmp_path):
    docs = [[100 + i] for i in range(37)]
    ds = make_dataset(tmp_path / "d", docs)
    s = PackedStream([Source("d", ds, 1.0)], 3, 9, BOS, EOS)  # each doc is [bos, x, eos]
    seen = [s.next_sequence()[1] for _ in range(37)]
    assert sorted(seen) == [100 + i for i in range(37)]
    assert s.epoch == [1]
    nxt = [s.next_sequence()[1] for _ in range(37)]
    assert sorted(nxt) == sorted(seen) and nxt != seen  # reshuffled next epoch


def test_stream_rejects_mismatched_state(data):
    a, b, _ = data
    s = PackedStream([Source("text", a, 0.7)], 32, 5, BOS, EOS)
    other = PackedStream([Source("text", a, 0.5)], 32, 5, BOS, EOS)
    with pytest.raises(ValueError):
        other.load_state_dict(s.state_dict())


def test_fixed_eval_batches_identical(data):
    v = data[2]
    a = fixed_eval_batches(v, 10, 32, 4, BOS, EOS)
    b = fixed_eval_batches(v, 10, 32, 4, BOS, EOS)
    assert [x.shape[0] for x in a] == [4, 4, 2]
    assert all(torch.equal(x, y) for x, y in zip(a, b, strict=True))


# ----------------------------------------------------------------------------- gates


def test_overfits_one_tiny_batch(tmp_path, smoke_model_cfg):
    doc = np.random.default_rng(0).integers(9, 8192, 30).tolist()
    ds = make_dataset(tmp_path / "one", [doc])  # [bos] + 30 + [eos] == one 32-token sequence
    cfg = resolved(
        smoke_model_cfg,
        max_steps=150,
        effective_batch_tokens=64,
        micro_batch_size=2,
        eval_every_steps=1000,
        checkpoint_every_steps=1000,
        log_every_steps=1000,
        optimizer={"peak_lr": 3e-3, "weight_decay": 0.0},
        scheduler={"name": "constant", "warmup_fraction": 0.0},
    )
    stream = PackedStream([Source("one", ds, 1.0)], 32, 0, BOS, EOS)
    t = Trainer(cfg, tmp_path / "run", stream, {}, "cpu")
    first = t.train_step()["loss"]
    for _ in range(149):
        last = t.train_step()["loss"]
    assert first > 8.0
    assert last < 0.05, last


def test_gradient_accumulation_matches_large_batch(tmp_path, smoke_model_cfg, data):
    big = make_trainer(resolved(smoke_model_cfg, micro_batch_size=8), tmp_path / "a", data)
    small = make_trainer(resolved(smoke_model_cfg, micro_batch_size=2), tmp_path / "b", data)
    assert big.accum == 1 and small.accum == 4
    loss_b, tok_b = big.accumulate_gradients()
    loss_s, tok_s = small.accumulate_gradients()
    assert tok_b == tok_s == 256
    assert loss_b == pytest.approx(loss_s, rel=1e-6)
    # Declared tolerance: fp32 summation order differs, so compare relative error per tensor.
    worst = 0.0
    for p, q in zip(big.model.parameters(), small.model.parameters(), strict=True):
        worst = max(worst, ((p.grad - q.grad).norm() / p.grad.norm().clamp_min(1e-12)).item())
    assert worst < 1e-5, worst
    # Same number of tokens and the same sequences reach the optimizer either way.
    assert big.stream.digest == small.stream.digest


def test_resume_matches_uninterrupted_run(tmp_path, smoke_model_cfg, data):
    cfg = resolved(smoke_model_cfg)
    full = make_trainer(cfg, tmp_path / "full", data)
    res_full = full.train()
    assert res_full.steps == 6

    ckpt3 = checkpoint_dir(tmp_path / "full", 3)
    resumed = make_trainer(cfg, tmp_path / "resumed", data)
    res = resumed.train(resume_from=ckpt3)

    assert res.steps == 6 and res.tokens == res_full.tokens
    sd_a, sd_b = full.model.state_dict(), resumed.model.state_dict()
    for k in sd_a:
        assert torch.allclose(sd_a[k], sd_b[k], atol=1e-7), k
    oa, ob = full.optimizer.state_dict()["state"], resumed.optimizer.state_dict()["state"]
    for k in oa:
        for name in ("exp_avg", "exp_avg_sq"):
            assert torch.allclose(oa[k][name], ob[k][name], atol=1e-9)
    assert full.stream.digest == resumed.stream.digest
    assert res.final_eval == pytest.approx(res_full.final_eval, rel=1e-6)

    def losses(run):
        rows = [json.loads(x) for x in (tmp_path / run / "metrics.jsonl").read_text().splitlines()]
        return {r["step"]: r["loss"] for r in rows if r["event"] == "train"}

    la, lb = losses("full"), losses("resumed")
    for step in (4, 5, 6):
        assert la[step] == pytest.approx(lb[step], rel=1e-6)


def test_two_fresh_runs_are_identical(tmp_path, smoke_model_cfg, data):
    cfg = resolved(smoke_model_cfg, max_steps=3, checkpoint_every_steps=3)
    a = make_trainer(cfg, tmp_path / "a", data)
    b = make_trainer(cfg, tmp_path / "b", data)
    a.train()
    b.train()
    for p, q in zip(a.model.parameters(), b.model.parameters(), strict=True):
        assert torch.equal(p, q)


# ----------------------------------------------------------------------------- checkpoints


def test_checkpoint_contents_and_latest(tmp_path, smoke_model_cfg, data):
    t = make_trainer(resolved(smoke_model_cfg), tmp_path / "run", data)
    t.train()
    assert latest_checkpoint(tmp_path / "run") == checkpoint_dir(tmp_path / "run", 6)
    m = verify_checkpoint(checkpoint_dir(tmp_path / "run", 3))
    assert m["step"] == 3 and m["config_hash"].startswith("sha256:")
    assert {"model/model.safetensors", "model/config.json", "training_state.pt"} <= set(m["files"])
    model, state, _ = load_checkpoint(checkpoint_dir(tmp_path / "run", 3))
    assert state["step"] == 3 and state["tokens"] == 3 * 256
    manifest = json.loads((tmp_path / "run" / "run_manifest.json").read_text())
    assert manifest["status"] == "completed" and manifest["estimated_cost_usd"] == 0.0


def test_checkpoints_never_overwritten(tmp_path, smoke_model_cfg, data):
    t = make_trainer(resolved(smoke_model_cfg, max_steps=3), tmp_path / "run", data)
    t.train()
    with pytest.raises(CheckpointError, match="never overwritten"):
        t.save()


def test_tampered_checkpoint_rejected(tmp_path, smoke_model_cfg, data):
    t = make_trainer(resolved(smoke_model_cfg, max_steps=3), tmp_path / "run", data)
    t.train()
    path = checkpoint_dir(tmp_path / "run", 3) / "training_state.pt"
    path.write_bytes(path.read_bytes() + b"x")
    with pytest.raises(CheckpointError, match="checksum"):
        load_checkpoint(path.parent)


def test_resume_refuses_different_config(tmp_path, smoke_model_cfg, data):
    make_trainer(resolved(smoke_model_cfg, max_steps=3), tmp_path / "run", data).train()
    other = make_trainer(resolved(smoke_model_cfg, max_steps=3, seed=1), tmp_path / "o", data)
    with pytest.raises(ValueError, match="different config"):
        other.resume(checkpoint_dir(tmp_path / "run", 3))


# ----------------------------------------------------------------------------- guards


def test_nonfinite_loss_stops_run(tmp_path, smoke_model_cfg, data):
    t = make_trainer(resolved(smoke_model_cfg), tmp_path / "run", data)
    original = t.model.forward

    def nan_forward(*a, **k):
        out = original(*a, **k)
        out.loss = out.loss * math.nan
        return out

    t.model.forward = nan_forward
    with pytest.raises(NonFiniteError):
        t.train()
    manifest = json.loads((tmp_path / "run" / "run_manifest.json").read_text())
    assert manifest["status"] == "failed"


def test_paid_compute_requires_budget_guard(tmp_path, smoke_model_cfg, data):
    budget = {
        "paid_compute_allowed": True, "max_spend_usd": 5, "provider": "x", "gpu_type": "y",
        "max_runtime_hours": 1, "hourly_rate_usd": 0.89, "shutdown_rule": "z",
        "approved_by": "a", "approved_at": "b",
    }  # fmt: skip
    with pytest.raises(ValueError, match="budget guard"):
        make_trainer(resolved(smoke_model_cfg, budget=budget), tmp_path / "run", data)


def test_builder_enforces_splits_and_tokenizer(tmp_path, smoke_model_cfg):
    from microreason.tokenizer import MicroTokenizer
    from microreason.train.pretrain import build_trainer

    tok_dir = Path(__file__).resolve().parents[1] / "artifacts" / "tokenizer" / "v1"
    sha = MicroTokenizer.load(tok_dir).sha256

    def ds(name, split, tok_sha=sha):
        write_tokenized(tmp_path / name, iter(random_docs(4, 20)),
                        {"name": name, "split": split, "tokenizer_sha256": tok_sha})  # fmt: skip
        return str(tmp_path / name)

    train, val, wrong = ds("tr", "train"), ds("va", "val"), ds("wr", "train", "0" * 64)

    def cfg(mix, valset):
        return resolved(smoke_model_cfg, tokenizer=str(tok_dir),
                        mixture=[{"dataset": mix, "weight": 1}],
                        validation=[{"dataset": valset, "sequences": 4}])  # fmt: skip

    t = build_trainer(cfg(train, val), tmp_path / "ok", "cpu")
    assert t.provenance["tokenizer_sha256"] == sha
    with pytest.raises(ValueError, match="split"):
        build_trainer(cfg(val, val), tmp_path / "a", "cpu")  # val data used for training
    with pytest.raises(ValueError, match="split"):
        build_trainer(cfg(train, train), tmp_path / "b", "cpu")  # train data used for eval
    with pytest.raises(ValueError, match="different tokenizer"):
        build_trainer(cfg(wrong, val), tmp_path / "c", "cpu")
