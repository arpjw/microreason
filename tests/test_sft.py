"""SFT stage: fixed selection, masked padded batches, exact accumulation and resume."""

import json
from pathlib import Path

import pytest
import torch
from pydantic import ValidationError
from test_generators import small_cfg

from microreason.checkpoint import checkpoint_dir, save_checkpoint
from microreason.config import ResolvedConfig, StageConfig, load_model_config
from microreason.data import build
from microreason.prompts import IGNORE_INDEX, build_example
from microreason.tokenizer import MicroTokenizer
from microreason.train.sft import SFTStream, build_sft_trainer, mode_for, pad_examples

ROOT = Path(__file__).resolve().parents[1]
TOK = ROOT / "artifacts" / "tokenizer" / "v1"


@pytest.fixture(scope="module")
def tok():
    return MicroTokenizer.load(TOK)


def test_mode_assignment_is_fixed_and_balanced():
    ids = [f"rec_{i}" for i in range(20_000)]
    modes = [mode_for(i, 0.75) for i in ids]
    assert modes == [mode_for(i, 0.75) for i in ids]
    assert abs(modes.count("show_work") / len(ids) - 0.75) < 0.01
    assert all(mode_for(i, 1.0) == "show_work" for i in ids[:100])


def test_padding_masks_labels_and_attention(tok):
    a = build_example(tok, "Compute: 1 + 1", ["Add: 1 + 1 = 2"], "2", "show_work")
    b = build_example(tok, "Compute: 2 * 3", [], "6", "direct")
    batch = pad_examples([a, b], tok.pad_id)
    n_b = len(b.input_ids)
    assert batch["input_ids"].shape[1] == len(a.input_ids) > n_b
    assert torch.all(batch["labels"][1, n_b:] == IGNORE_INDEX)
    assert torch.all(batch["attention_mask"][1, n_b:] == 0)
    assert batch["labels"][1, :n_b].tolist() == b.labels


def _items(tok, n):
    return [(f"id{i}", build_example(tok, f"Compute: {i} + 1", [f"Add: {i} + 1 = {i + 1}"],
                                     str(i + 1), "show_work")) for i in range(n)]  # fmt: skip


def test_stream_epochs_and_resume(tok):
    s = SFTStream(_items(tok, 10), 0, tok.pad_id, "t")
    first = [s.next_batch(5)["input_ids"] for _ in range(2)]
    assert s.epoch == 0 and s.pos == 10
    state = json.loads(json.dumps(s.state_dict()))
    want = s.next_batch(7)["input_ids"]
    assert s.epoch == 1
    r = SFTStream(_items(tok, 10), 0, tok.pad_id, "t")
    r.load_state_dict(state)
    assert torch.equal(r.next_batch(7)["input_ids"], want)
    seen = torch.cat([x[:, :9] for x in first])
    assert len({tuple(row.tolist()) for row in seen}) == 10  # every example once per epoch
    with pytest.raises(ValueError):
        SFTStream(_items(tok, 9), 0, tok.pad_id, "t").load_state_dict(state)


@pytest.fixture
def setup(tmp_path, tok):
    data = small_cfg(tmp_path / "data")
    build.write_dataset(data, build.build_dataset(data))
    model_cfg = load_model_config(ROOT / "configs" / "model_smoke.yaml")
    from microreason.model import build_model

    torch.manual_seed(0)
    parent = save_checkpoint(tmp_path / "parent", 5, build_model(model_cfg),
                             torch.optim.AdamW(build_model(model_cfg).parameters()),
                             {"step": 5}, {"config_hash": "sha256:parent"})  # fmt: skip
    return data, model_cfg, parent


def sft_cfg(data, model_cfg, parent, **over):
    base = dict(
        stage="sft", model="configs/model_smoke.yaml", tokenizer=str(TOK), parent=str(parent),
        sft={"train_manifest": str(build.manifest_path(data.manifest_dir, "v1", "train")),
             "val_manifest": str(build.manifest_path(data.manifest_dir, "v1", "val")),
             "num_examples": 100, "show_work_fraction": 0.75, "val_examples_per_format": 8},
        seed=0, precision="fp32", sequence_length=256, effective_batch_tokens=256 * 8,
        micro_batch_size=4, max_examples=200, log_every_steps=1, eval_every_steps=5,
        checkpoint_every_steps=5, optimizer={"peak_lr": 1e-3},
    )  # fmt: skip
    base.update(over)
    return ResolvedConfig(stage=StageConfig.model_validate(base), model=model_cfg)


def test_sft_config_validation(setup):
    data, model_cfg, parent = setup
    with pytest.raises(ValidationError, match="sft section"):
        sft_cfg(data, model_cfg, parent, sft=None)
    with pytest.raises(ValidationError, match="only valid for sft"):
        sft_cfg(data, model_cfg, parent, stage="pretrain", max_steps=1)


def test_sft_accumulation_matches_large_batch_with_variable_targets(tmp_path, setup):
    data, model_cfg, parent = setup
    big = build_sft_trainer(sft_cfg(data, model_cfg, parent, micro_batch_size=8), tmp_path / "a",
                            "cpu")  # fmt: skip
    small = build_sft_trainer(sft_cfg(data, model_cfg, parent, micro_batch_size=2),
                              tmp_path / "b", "cpu")  # fmt: skip
    lb, _ = big.accumulate_gradients()
    ls, _ = small.accumulate_gradients()
    assert lb == pytest.approx(ls, rel=1e-5)
    pairs = zip(big.model.parameters(), small.model.parameters(), strict=True)
    worst = max(
        ((p.grad - q.grad).norm() / p.grad.norm().clamp_min(1e-12)).item() for p, q in pairs
    )
    assert worst < 1e-5, worst


def test_sft_trains_resumes_and_records_provenance(tmp_path, setup):
    data, model_cfg, parent = setup
    cfg = sft_cfg(data, model_cfg, parent, max_examples=80)  # 10 steps of 8 examples
    full = build_sft_trainer(cfg, tmp_path / "full", "cpu")
    res = full.train()
    assert res.steps == 10 and full.examples == 80
    m = json.loads((tmp_path / "full" / "run_manifest.json").read_text())
    sft = m["provenance"]["sft_data"]
    assert sft["examples"] == 100 and sum(sft["by_mode"].values()) == 100
    assert m["provenance"]["parent_step"] == 5
    assert set(res.final_eval) == {"sft_val_show_work", "sft_val_direct"}
    rows = [json.loads(x) for x in (tmp_path / "full" / "metrics.jsonl").read_text().splitlines()]
    losses = [r["loss"] for r in rows if r["event"] == "train"]
    assert losses[-1] < losses[0]

    resumed = build_sft_trainer(cfg, tmp_path / "resumed", "cpu")
    resumed.train(resume_from=checkpoint_dir(tmp_path / "full", 5))
    for p, q in zip(full.model.parameters(), resumed.model.parameters(), strict=True):
        assert torch.allclose(p, q, atol=1e-6)
    assert full.stream.digest == resumed.stream.digest and resumed.examples == 80
