from pathlib import Path

import pytest
from pydantic import ValidationError

from microreason.data.sources import TextSource, load_sources

CONFIG = Path(__file__).resolve().parents[1] / "configs" / "data_sources.yaml"


def test_shipped_sources_validate():
    cfg = load_sources(CONFIG)
    (cosmo,) = cfg.sources
    assert cosmo.license == "apache-2.0"
    assert 1_000_000_000 < cosmo.total_bytes() < 1_500_000_000
    assert "auto_math_text" in cosmo.excluded_subsets


def _source(**overrides):
    base = load_sources(CONFIG).sources[0].model_dump()
    base.update(overrides)
    return TextSource.model_validate(base)


def test_branch_revision_rejected():
    with pytest.raises(ValidationError, match="40-character"):
        _source(revision="main")


def test_unapproved_license_rejected():
    with pytest.raises(ValidationError, match="approved"):
        _source(license="cc-by-sa-3.0")


def test_file_from_excluded_subset_rejected():
    files = [{"path": "data/auto_math_text/x.parquet", "size_bytes": 1, "sha256": "a" * 64}]
    with pytest.raises(ValidationError, match="excluded subset"):
        _source(files=files)
