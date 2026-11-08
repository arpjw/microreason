from pathlib import Path

import pytest
from pydantic import ValidationError

from microreason.config import (
    BudgetConfig,
    ModelConfig,
    StageConfig,
    load_model_config,
    load_stage_config,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "configs"


def _stage(**overrides):
    base = dict(
        stage="pretrain",
        model="configs/model_43m.yaml",
        sequence_length=1024,
        effective_batch_tokens=65536,
        max_train_tokens=1000,
        optimizer={"peak_lr": 3e-4},
    )
    base.update(overrides)
    return StageConfig.model_validate(base)


def test_full_model_parameter_count_matches_spec():
    model = load_model_config(CONFIGS / "model_43m.yaml")
    assert model.parameter_count() == 42_742_272


def test_smoke_model_is_2_to_5_million_params():
    model = load_model_config(CONFIGS / "model_smoke.yaml")
    assert 2_000_000 <= model.parameter_count() <= 5_000_000


def test_untied_embeddings_add_lm_head():
    tied = load_model_config(CONFIGS / "model_43m.yaml")
    untied = tied.model_copy(update={"tie_embeddings": False})
    assert untied.parameter_count() - tied.parameter_count() == 8192 * 512


def test_head_shape_must_match_width():
    with pytest.raises(ValidationError, match="d_model"):
        ModelConfig(
            name="bad",
            n_layers=1,
            d_model=512,
            n_heads=8,
            n_kv_heads=8,
            head_dim=32,
            mlp_hidden=1408,
            vocab_size=8192,
            max_seq_len=1024,
        )


def test_unknown_keys_rejected():
    with pytest.raises(ValidationError):
        _stage(learning_rate=1e-3)


@pytest.mark.parametrize(
    "name", ["smoke.yaml", "pretrain.yaml", "midtrain.yaml", "midtrain_uniform.yaml"]
)
def test_shipped_stage_configs_validate_and_disallow_spend(name):
    resolved = load_stage_config(CONFIGS / name)
    assert resolved.stage.budget.paid_compute_allowed is False
    assert resolved.stage.budget.max_spend_usd == 0


def test_config_hash_is_stable_and_sensitive():
    a = load_stage_config(CONFIGS / "pretrain.yaml")
    b = load_stage_config(CONFIGS / "pretrain.yaml")
    assert a.config_hash() == b.config_hash()
    assert a.config_hash().startswith("sha256:")
    changed = a.model_copy(update={"stage": a.stage.model_copy(update={"seed": 7})})
    assert changed.config_hash() != a.config_hash()


def test_budget_defaults_to_zero_spend():
    b = BudgetConfig()
    assert b.paid_compute_allowed is False
    assert b.max_spend_usd == 0


def test_nonzero_spend_without_authorization_rejected():
    with pytest.raises(ValidationError, match="max_spend_usd must be 0"):
        BudgetConfig(max_spend_usd=10)


def test_paid_compute_requires_all_authorization_fields():
    with pytest.raises(ValidationError, match="provider"):
        BudgetConfig(paid_compute_allowed=True, max_spend_usd=50)


def test_paid_compute_accepted_when_fully_specified():
    b = BudgetConfig(
        paid_compute_allowed=True,
        max_spend_usd=50,
        provider="runpod",
        gpu_type="RTX 4090",
        max_runtime_hours=10,
        hourly_rate_usd=0.89,
        shutdown_rule="terminate at cap",
        approved_by="owner",
        approved_at="2026-11-07",
    )
    assert b.max_spend_usd == 50


def test_batch_must_be_multiple_of_sequence_length():
    with pytest.raises(ValidationError, match="multiple"):
        _stage(effective_batch_tokens=1000)


def test_requires_some_stopping_limit():
    with pytest.raises(ValidationError, match="at least one"):
        _stage(max_train_tokens=None)


def test_later_stages_require_parent():
    with pytest.raises(ValidationError, match="parent"):
        _stage(stage="sft")
    assert _stage(stage="sft", parent="artifacts/v1-math").parent == "artifacts/v1-math"


def test_sequence_length_cannot_exceed_model_context(tmp_path):
    cfg = tmp_path / "too_long.yaml"
    cfg.write_text(
        (CONFIGS / "smoke.yaml")
        .read_text()
        .replace("model: configs/model_smoke.yaml", f"model: {CONFIGS / 'model_smoke.yaml'}")
        .replace("sequence_length: 256", "sequence_length: 1024")
        .replace("micro_batch_size: 16", "micro_batch_size: 4")
    )
    with pytest.raises(ValidationError, match="max_seq_len"):
        load_stage_config(cfg)
