import json
import subprocess
import sys
from pathlib import Path

import pytest

from microreason.cli import EXIT_CONFIG_ERROR, EXIT_NOT_IMPLEMENTED, EXIT_OK, main

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "configs"


def test_help_runs_as_module():
    result = subprocess.run(
        [sys.executable, "-m", "microreason.cli", "--help"], capture_output=True, text=True
    )
    assert result.returncode == 0
    for cmd in ("config", "model", "data", "tokenizer", "train", "eval", "compare", "serve"):
        assert cmd in result.stdout


def test_config_validate_prints_hash(capsys):
    assert main(["config", "validate", "--config", str(CONFIGS / "pretrain.yaml")]) == EXIT_OK
    out = json.loads(capsys.readouterr().out)
    assert out["config_hash"].startswith("sha256:")
    assert out["parameter_count_analytic"] == 42_742_272


def test_config_validate_bad_file(tmp_path, capsys):
    bad = tmp_path / "bad.yaml"
    bad.write_text("stage: pretrain\n")
    assert main(["config", "validate", "--config", str(bad)]) == EXIT_CONFIG_ERROR


def test_model_inspect(capsys):
    assert main(["model", "inspect", "--config", str(CONFIGS / "model_smoke.yaml")]) == EXIT_OK
    out = json.loads(capsys.readouterr().out)
    assert out["parameter_count_instantiated"] == out["parameter_count_analytic"] == 3_344_064


def test_train_refuses_unimplemented_stage(tmp_path):
    # Must never launch a real run from the test suite: use a stage that is not built yet.
    cfg = tmp_path / "sft.yaml"
    cfg.write_text(
        (CONFIGS / "smoke.yaml")
        .read_text()
        .replace("stage: smoke", "stage: sft\nparent: artifacts/v1-math")
        .replace("model: configs/model_smoke.yaml", f"model: {CONFIGS / 'model_smoke.yaml'}")
    )
    assert main(["train", "--config", str(cfg), "--run-dir", str(tmp_path / "r")]) == (
        EXIT_NOT_IMPLEMENTED
    )
    assert not (tmp_path / "r").exists()


def test_paid_run_needs_pod_id_and_billing_start(tmp_path, monkeypatch):
    monkeypatch.delenv("RUNPOD_POD_ID", raising=False)
    monkeypatch.delenv("MICROREASON_BILLING_START", raising=False)
    cfg = tmp_path / "paid.yaml"
    paid = """budget:
  paid_compute_allowed: true
  max_spend_usd: 5
  provider: runpod
  gpu_type: RTX 4090
  max_runtime_hours: 1
  hourly_rate_usd: 0.89
  shutdown_rule: terminate at cap
  approved_by: nobody
  approved_at: never"""
    text = (CONFIGS / "smoke.yaml").read_text()
    text = text[: text.index("budget:")] + paid + "\n"
    cfg.write_text(text.replace("model: configs/model_smoke.yaml",
                                f"model: {CONFIGS / 'model_smoke.yaml'}"))  # fmt: skip
    assert main(["train", "--config", str(cfg), "--run-dir", str(tmp_path / "r")]) == (
        EXIT_CONFIG_ERROR
    )
    assert not (tmp_path / "r").exists()


@pytest.mark.parametrize(
    "argv",
    [
        ["serve", "--checkpoint", "x"],
    ],
)
def test_stubs_exit_not_implemented(argv, capsys):
    assert main(argv) == EXIT_NOT_IMPLEMENTED
    assert "not implemented" in capsys.readouterr().err
