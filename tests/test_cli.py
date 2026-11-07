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
    assert out["parameter_count_instantiated"] is None


def test_train_validates_then_refuses():
    assert main(["train", "--config", str(CONFIGS / "smoke.yaml")]) == EXIT_NOT_IMPLEMENTED


@pytest.mark.parametrize(
    "argv",
    [
        ["data", "generate", "--config", "x", "--out", "y"],
        ["data", "audit", "--manifest", "x"],
        ["tokenizer", "train", "--config", "x"],
        ["eval", "--suite", "x", "--checkpoint", "y"],
        ["compare", "--runs", "x"],
        ["serve", "--checkpoint", "x"],
    ],
)
def test_stubs_exit_not_implemented(argv, capsys):
    assert main(argv) == EXIT_NOT_IMPLEMENTED
    assert "not implemented" in capsys.readouterr().err
