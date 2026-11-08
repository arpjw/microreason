"""RL-stage branches: advantages, GRPO/DPO gradient direction, clipping, pool partitions, gate,
and tiny end-to-end GRPO / DPO / time-matched extra-SFT runs on CPU."""

import json
from pathlib import Path

import pytest
import torch
from test_generators import small_cfg
from test_stream import stream_cfg

from microreason.checkpoint import load_checkpoint, save_checkpoint
from microreason.config import ResolvedConfig, StageConfig, load_model_config
from microreason.data import build
from microreason.model import build_model
from microreason.prompts import prompt_ids
from microreason.rl import dpo as dpo_mod
from microreason.rl.config import DPOConfig, GRPOConfig, ResolvedRL
from microreason.rl.gate import decide, run_gate
from microreason.rl.grpo import GRPOTrainer, group_advantages, grpo_loss
from microreason.rl.pool import PARTITIONS, load_partition, partition_of
from microreason.rl.rollout import Completion, build_sequences, token_logprobs
from microreason.tokenizer import MicroTokenizer

ROOT = Path(__file__).resolve().parents[1]
TOK = ROOT / "artifacts" / "tokenizer" / "v1"
SMOKE = ROOT / "configs" / "model_smoke.yaml"


@pytest.fixture(scope="module")
def tok():
    return MicroTokenizer.load(TOK)


@pytest.fixture(scope="module")
def model_cfg():
    return load_model_config(SMOKE)


@pytest.fixture
def parent(tmp_path, model_cfg, tok):
    torch.manual_seed(0)
    m = build_model(model_cfg)
    return save_checkpoint(tmp_path / "parent", 1, m, torch.optim.AdamW(m.parameters()),
                           {"step": 1}, {"config_hash": "sha256:parent",
                                         "tokenizer_sha256": tok.sha256})  # fmt: skip


@pytest.fixture
def records(tmp_path):
    data = small_cfg(tmp_path / "data")
    build.write_dataset(data, build.build_dataset(data))
    m = json.loads(build.manifest_path(data.manifest_dir, "v1", "train").read_text())
    return [r.model_dump(mode="json") for r in build.load_split(m)]


# ----------------------------------------------------------------------------- advantages


def test_group_advantages_signs_and_zero_variance():
    r = torch.tensor([[1.0, 0, 0, 0], [1, 1, 1, 1], [0, 0, 0, 0], [1, 1, 0, 0]])
    a = group_advantages(r)
    assert a[0, 0] > 0 and (a[0, 1:] < 0).all()
    assert torch.allclose(a[0].sum(), torch.tensor(0.0), atol=1e-5)
    assert (a[1] == 0).all() and (a[2] == 0).all()  # no invented signal
    assert torch.allclose(a[3], torch.tensor([1.0, 1, -1, -1]), atol=1e-4)


def test_grpo_loss_kl_zero_at_reference_and_clipping_blocks_gradient():
    lp = torch.log(torch.full((2, 3), 0.5))
    mask = torch.ones(2, 3, dtype=torch.bool)
    adv = torch.tensor([1.0, -1.0])
    _, st = grpo_loss(lp, lp, lp, mask, adv, 0.2, 0.04)
    assert st["kl"] == pytest.approx(0.0, abs=1e-7) and st["clip_fraction"] == 0.0
    # ratio 1.5 > 1 + eps with positive advantage: the clipped branch is active, no gradient
    new = (lp + torch.log(torch.tensor(1.5))).clone().requires_grad_(True)
    loss, st = grpo_loss(new, lp, new.detach(), mask, torch.tensor([1.0, 1.0]), 0.2, 0.0)
    loss.backward()
    assert st["clip_fraction"] == 1.0 and torch.count_nonzero(new.grad) == 0


def _two_response_batch(tok):
    p = prompt_ids(tok, "Compute: 2 + 3", "show_work")
    good = [*tok.encode("[answer]"), *tok.encode("5"), tok.eos_id]
    bad = [*tok.encode("[answer]"), *tok.encode("7"), tok.eos_id]
    return build_sequences([p, p], [good, bad], tok.pad_id)


def _grpo_sgd_step(model, batch, adv, lr=1e-3):
    opt = torch.optim.SGD(model.parameters(), lr=lr)
    new = token_logprobs(model, batch)
    loss, _ = grpo_loss(new, new.detach(), new.detach(), batch["response_mask"], adv, 0.2, 0.0)
    loss.backward()
    opt.step()


def test_grpo_update_moves_toward_the_rewarded_response(tok, model_cfg):
    """Analytic check (spec §6 Stage 5): two responses with rewards 1 and 0 get advantages +1 and
    -1. A small gradient step must raise the rewarded response's log probability relative to the
    other's, and a lone response must move up for A > 0 and down for A < 0."""
    batch = _two_response_batch(tok)
    adv = group_advantages(torch.tensor([[1.0, 0.0]])).flatten()
    assert adv[0] > 0 > adv[1]
    torch.manual_seed(1)
    model = build_model(model_cfg)
    with torch.no_grad():
        before = token_logprobs(model, batch).sum(dim=1)
    _grpo_sgd_step(model, batch, adv)
    with torch.no_grad():
        after = token_logprobs(model, batch).sum(dim=1)
    assert (after[0] - after[1]) > (before[0] - before[1])
    one = {k: v[:1] for k, v in batch.items()}
    for sign in (1.0, -1.0):
        torch.manual_seed(1)
        model = build_model(model_cfg)
        with torch.no_grad():
            b0 = token_logprobs(model, one).sum()
        _grpo_sgd_step(model, one, torch.tensor([sign]))
        with torch.no_grad():
            b1 = token_logprobs(model, one).sum()
        assert (b1 - b0) * sign > 0


def test_dpo_update_prefers_chosen(tok, model_cfg):
    torch.manual_seed(2)
    model = build_model(model_cfg)
    batch = _two_response_batch(tok)
    with torch.no_grad():
        ref = token_logprobs(model, batch).sum(dim=1)
    opt = torch.optim.SGD(model.parameters(), lr=0.05)
    lp = token_logprobs(model, batch).sum(dim=1)
    loss, st = dpo_mod.dpo_loss(lp[:1], lp[1:], ref[:1], ref[1:], beta=0.1)
    assert loss.item() == pytest.approx(0.6931, abs=1e-3) and st["margin"] == pytest.approx(0)
    loss.backward()
    opt.step()
    with torch.no_grad():
        after = token_logprobs(model, batch).sum(dim=1)
    assert (after[0] - ref[0]) - (after[1] - ref[1]) > 0


def test_response_mask_covers_exactly_the_completion(tok):
    b = build_sequences([[1, 2, 3], [4, 5]], [[6, 7], [8, 9, 10]], pad_id=0)
    assert b["response_mask"].sum(dim=1).tolist() == [2, 3]
    # row 0: targets at ids positions 3,4 -> logit positions 2,3
    assert b["response_mask"][0].tolist() == [False, False, True, True]
    assert b["attention_mask"][0].tolist() == [1, 1, 1, 1, 1]


# ----------------------------------------------------------------------------- pool / gate


def test_partitions_are_deterministic_and_proportioned():
    ids = [f"math_rl1_{i}" for i in range(40_000)]
    parts = [partition_of(i) for i in ids]
    assert parts == [partition_of(i) for i in ids]
    lo = 0.0
    for name, hi in PARTITIONS:
        assert abs(parts.count(name) / len(ids) - (hi - lo)) < 0.01
        lo = hi


def test_gate_rule():
    cfg = GRPOConfig(model=str(SMOKE))
    ok = {"mixed_group_fraction": 0.4, "malformed_fraction": 0.01, "pass_at_1": 0.8}
    assert decide(ok, cfg) == (True, [])
    passed, why = decide({**ok, "mixed_group_fraction": 0.05, "pass_at_1": 0.99}, cfg)
    assert not passed and len(why) == 2


def test_gate_runs_on_training_records(tmp_path, tok, model_cfg, parent, records):
    cfg = GRPOConfig(
        model=str(SMOKE),
        tokenizer=str(TOK),
        precision="fp32",
        sampling={"group_size": 2, "max_new_tokens": 6, "generation_batch_size": 8},
    )
    rep = run_gate(ResolvedRL(stage=cfg, model=model_cfg), str(parent), tmp_path / "g.json",
                   "cpu", records=records[:6])  # fmt: skip
    o = rep["overall"]
    assert o["groups"] == 6 and o["completions"] == 12
    assert rep["passed"] is False  # an untrained model never answers
    assert json.loads((tmp_path / "g.json").read_text())["failures"] == rep["failures"]


# ----------------------------------------------------------------------------- end to end


def test_grpo_tiny_run_writes_rollouts_checkpoint_and_manifest(tmp_path, model_cfg, parent,
                                                               records):  # fmt: skip
    cfg = GRPOConfig(model=str(SMOKE), tokenizer=str(TOK), parent=str(parent), precision="fp32",
                     sampling={"group_size": 2, "max_new_tokens": 6, "generation_batch_size": 8},
                     prompts_per_step=3, steps=2, warmup_steps=1, checkpoint_every_steps=1,
                     log_every_steps=1)  # fmt: skip
    run = tmp_path / "grpo"
    res = GRPOTrainer(ResolvedRL(stage=cfg, model=model_cfg), run, "cpu", pool=records).train()
    assert res["status"] == "completed" and res["updates"] == 4
    rows = [json.loads(x) for x in (run / "rollouts.jsonl").read_text().splitlines()]
    assert len(rows) == 2 * 3 * 2
    assert {r["policy_updates_before"] for r in rows} == {0, 2}
    assert all(len(r["old_logprobs"]) == len(r["token_ids"]) == r["n_tokens"] for r in rows)
    man = json.loads((run / "run_manifest.json").read_text())
    assert man["status"] == "completed" and man["timing"]["train_wall_seconds"] > 0
    model, _, cm = load_checkpoint(man["final_checkpoint"])
    assert cm["stage"] == "grpo" and cm["tokenizer_sha256"]
    with pytest.raises(FileExistsError):
        GRPOTrainer(ResolvedRL(stage=cfg, model=model_cfg), run, "cpu", pool=records)


def test_dpo_tiny_run_with_stubbed_samples(tmp_path, tok, model_cfg, parent, records, monkeypatch):
    def fake_groups(model, tok_, recs, **kw):
        out = []
        for r in recs:
            good = [*tok.encode("[answer]" + r["final_answer"] + "[/answer]"), tok.eos_id]
            out.append(
                [
                    Completion(good, True, "correct", "match", 1.0),
                    Completion(good[:-1], False, "truncated", "no_eos", 0.0),
                ]
            )
        return out

    monkeypatch.setattr(dpo_mod, "sample_groups", fake_groups)
    cfg = DPOConfig(model=str(SMOKE), tokenizer=str(TOK), parent=str(parent), precision="fp32",
                    max_pair_prompts=40, target_pairs=30, batch_pairs=8, heldout_fraction=0.3,
                    lr=1e-3)  # fmt: skip
    res = dpo_mod.DPOTrainer(ResolvedRL(stage=cfg, model=model_cfg), tmp_path / "dpo", "cpu",
                             pool=records).train()  # fmt: skip
    assert res["status"] == "completed"
    assert res["pairs"]["pairs"] == 30 and res["pairs"]["train"] + res["pairs"]["heldout"] == 30
    assert res["heldout_before"]["implicit_reward_accuracy"] == 0.0  # policy == reference
    assert res["steps"] >= 1
    assert (tmp_path / "dpo" / "pairs.jsonl").exists()


def test_extra_sft_from_stream_partition_stops_at_wall_limit(tmp_path, model_cfg, parent):

    from microreason.data.stream import build_stream
    from microreason.train.sft import build_sft_trainer

    ev = small_cfg(tmp_path / "eval")
    build.write_dataset(ev, build.build_dataset(ev))
    build.freeze(ev.manifest_dir)
    scfg = stream_cfg(tmp_path, ev, "rl", targets={"add_ab": 400, "mul_sub_mul": 400})
    build_stream(scfg)
    recs, _ = load_partition(scfg.manifest_path, "sft_extra")
    assert recs and all(partition_of(r["id"]) == "sft_extra" for r in recs)
    stage = StageConfig.model_validate(dict(
        stage="extra_sft", model=str(SMOKE), tokenizer=str(TOK), parent=str(parent),
        sft={"train_manifest": scfg.manifest_path, "stream_partition": "sft_extra",
             "val_manifest": str(build.manifest_path(ev.manifest_dir, "v1", "val")),
             "num_examples": 100, "val_examples_per_format": 4},
        seed=0, precision="fp32", sequence_length=256, effective_batch_tokens=256 * 4,
        micro_batch_size=4, max_examples=10_000_000, max_wall_seconds=1.0, log_every_steps=1,
        eval_every_steps=10_000, checkpoint_every_steps=10_000,
        optimizer={"peak_lr": 1e-4}, scheduler={"name": "constant", "warmup_fraction": 0.0},
    ))  # fmt: skip
    resolved = ResolvedConfig(stage=stage, model=model_cfg)
    res = build_sft_trainer(resolved, tmp_path / "xsft", "cpu").train()
    man = json.loads((tmp_path / "xsft" / "run_manifest.json").read_text())
    assert man["status"] == "completed" and man["timing"]["stopped_at_wall_limit"] is True
    assert 0 < res.steps < resolved.stage.total_steps() and res.final_checkpoint is not None
    assert man["provenance"]["sft_data"]["selection_rule"].startswith("partition sft_extra")
