"""Schema for pinned, licensed external text sources (spec §4)."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Self

import yaml
from pydantic import BaseModel, ConfigDict, Field, PositiveInt, field_validator, model_validator

# Licenses already reviewed as permitting training and redistribution for this project.
APPROVED_LICENSES = frozenset({"apache-2.0", "mit", "cc-by-4.0"})

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_REVISION = re.compile(r"^[0-9a-f]{40}$")


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class SourceFile(_Strict):
    path: str
    size_bytes: PositiveInt
    sha256: str

    @field_validator("sha256")
    @classmethod
    def _hex(cls, v: str) -> str:
        if not _SHA256.match(v):
            raise ValueError("sha256 must be 64 lowercase hex characters")
        return v


class TextSource(_Strict):
    name: str
    hub_repo: str
    revision: str
    license: str
    license_url: str
    retrieved_metadata_on: str
    provenance: str
    text_field: str = "text"
    excluded_subsets: list[str] = Field(default_factory=list)
    files: list[SourceFile] = Field(min_length=1)

    @field_validator("revision")
    @classmethod
    def _pinned(cls, v: str) -> str:
        if not _REVISION.match(v):
            raise ValueError("revision must be a full 40-character commit hash, not a branch")
        return v

    @model_validator(mode="after")
    def _check(self) -> Self:
        if self.license not in APPROVED_LICENSES:
            raise ValueError(f"license '{self.license}' is not on the approved list")
        for f in self.files:
            parts = f.path.split("/")
            if len(parts) > 1 and parts[1] in self.excluded_subsets:
                raise ValueError(f"{f.path} belongs to excluded subset '{parts[1]}'")
        return self

    def total_bytes(self) -> int:
        return sum(f.size_bytes for f in self.files)


class SourcesConfig(_Strict):
    sources: list[TextSource] = Field(min_length=1)


def load_sources(path: str | Path) -> SourcesConfig:
    with Path(path).open("r", encoding="utf-8") as f:
        return SourcesConfig.model_validate(yaml.safe_load(f))
