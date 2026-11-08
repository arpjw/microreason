"""Paid-compute budget guard (spec §8): ceilings, projection stop, ledger, authorization."""

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from microreason.checkpoint import checkpoint_dir
from microreason.config import BudgetConfig, load_stage_config
from microreason.train.budget import STOP_CEILING, STOP_PROJECTION, BudgetGuard

CONFIGS = Path(__file__).resolve().parents[1] / "configs"

APPROVED = dict(
    paid_compute_allowed=True,
    max_spend_usd=5.0,
    provider="runpod",
    gpu_type="RTX 4090",
    max_runtime_hours=3.0,
    hourly_rate_usd=0.89,
    shutdown_rule="terminate",
    approved_by="owner",
    approved_at="2026-11-07",
    reserve_minutes=10,
)


class FakeClock:
    def __init__(self, t=0.0):
        self.t = t

    def __call__(self):
        return self.t


def guard(tmp_path, clock, **over):
    return BudgetGuard(BudgetConfig(**{**APPROVED, **over}), 0.0, "pod123",
                       tmp_path / "ledger.json", clock)  # fmt: skip


def test_ceiling_is_tighter_of_time_and_dollars():
    assert BudgetConfig(**APPROVED).ceiling_hours() == 3.0
    # $1 at $0.89/h buys 1.12 h, tighter than the 3 h runtime limit
    assert BudgetConfig(**{**APPROVED, "max_spend_usd": 1.0}).ceiling_hours() == pytest.approx(
        1 / 0.89
    )


@pytest.mark.parametrize("field", ["approved_by", "approved_at"])
@pytest.mark.parametrize("value", ["PENDING", "tbd", " TODO "])
def test_placeholder_approval_rejected(field, value):
    with pytest.raises(ValidationError, match="not authorized"):
        BudgetConfig(**{**APPROVED, field: value})


def test_unquoted_yaml_date_approval_accepted():
    import yaml

    b = BudgetConfig(**{**APPROVED, "approved_at": yaml.safe_load("d: 2026-11-07")["d"]})
    assert b.approved_at == "2026-11-07"


def test_paid_requires_hourly_rate():
    with pytest.raises(ValidationError, match="hourly_rate_usd"):
        BudgetConfig(**{**APPROVED, "hourly_rate_usd": None})


def test_shipped_pilot_config_is_not_yet_authorized():
    with pytest.raises(ValidationError, match="not authorized"):
        load_stage_config(CONFIGS / "pilot_43m.yaml")


def test_guard_allows_then_stops_at_ceiling(tmp_path):
    clock = FakeClock(1000)
    g = guard(tmp_path, clock)
    assert g.check(100, 10, 10) is None
    clock.t = 3 * 3600 - 600 - 50  # next 100 s step would eat into the 10-minute reserve
    assert g.check(100, 10, 10) == STOP_CEILING
    ledger = json.loads((tmp_path / "ledger.json").read_text())
    assert ledger["stop_reason"] == STOP_CEILING
    assert ledger["pod_id"] == "pod123" and ledger["approved_max_usd"] == 5.0
    assert ledger["estimated_accrued_usd"] == pytest.approx((3 * 3600 - 650) / 3600 * 0.89, 1e-3)


def test_guard_projection_stop_only_after_measurement_window(tmp_path):
    clock = FakeClock(600)
    g = guard(tmp_path, clock)
    # 200 steps x 100 s = 5.6 h of GPU time would cost about $5.1: over the $5 cap.
    assert g.check(100, 2, 200) is None  # still measuring
    assert g.check(100, 5, 200) == STOP_PROJECTION
    assert g.check(30, 5, 200) is None  # affordable at 30 s/step


def test_guard_refuses_exhausted_budget(tmp_path):
    with pytest.raises(RuntimeError, match="exhausted"):
        guard(tmp_path, FakeClock(3 * 3600))


def test_guard_needs_pod_id(tmp_path):
    with pytest.raises(ValueError, match="pod id"):
        BudgetGuard(BudgetConfig(**APPROVED), 0.0, "", tmp_path / "l.json", FakeClock())


def test_trainer_stops_and_checkpoints_at_ceiling(tmp_path):
    from test_training import make_dataset, random_docs, resolved

    from microreason.config import load_model_config
    from microreason.train.common import Trainer
    from microreason.train.data import PackedStream, Source

    ds = make_dataset(tmp_path / "d", random_docs(1, 200), name="d")
    cfg = resolved(
        load_model_config(CONFIGS / "model_smoke.yaml"),
        max_steps=1000,
        log_every_steps=1000,
        eval_every_steps=1000,
        checkpoint_every_steps=1000,
        budget=APPROVED,
    )
    clock = FakeClock(0)
    g = BudgetGuard(cfg.stage.budget, 0.0, "pod123", tmp_path / "run" / "budget_ledger.json",
                    clock)  # fmt: skip
    stream = PackedStream([Source("d", ds, 1.0)], 32, 0, 1, 2)
    t = Trainer(cfg, tmp_path / "run", stream, {}, "cpu", budget_guard=g)
    real_step = t.train_step

    def billed_step():
        clock.t += 600  # each step "costs" 10 minutes of billed time
        return real_step()

    t.train_step = billed_step
    res = t.train()
    # Before step k the elapsed time is 600(k-1); the run must stop once one more step plus the
    # 10-minute reserve would pass 3 h, i.e. after 17 steps.
    assert res.stop_reason == STOP_CEILING
    assert res.steps == 17
    assert res.final_checkpoint == checkpoint_dir(tmp_path / "run", 17)
    manifest = json.loads((tmp_path / "run" / "run_manifest.json").read_text())
    assert manifest["status"] == STOP_CEILING
    assert manifest["estimated_cost_usd"] == pytest.approx(17 * 600 / 3600 * 0.89, 1e-3)
    ledger = json.loads((tmp_path / "run" / "budget_ledger.json").read_text())
    assert ledger["stop_reason"] == STOP_CEILING
