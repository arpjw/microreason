"""Record and build-config schemas for generated math data (spec §4)."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, PositiveInt

Split = Literal[
    "train",
    "train_stream",
    "val",
    "test_id",
    "test_extrapolation",
    "test_heldout_template",
    "val_heldout_shape",
    "test_heldout_shape",
]
EVAL_SPLITS: tuple[Split, ...] = ("test_heldout_template", "test_extrapolation", "test_id", "val")
ALL_SPLITS: tuple[Split, ...] = (*EVAL_SPLITS, "train")
# Data v3 (generator 3.0, reports/protocol_v3.md): held-out *shapes*, with a development
# held-out set (val_heldout_shape) shape-disjoint from the sealed one (test_heldout_shape).
V3_EVAL_SPLITS: tuple[Split, ...] = (
    "test_heldout_shape",
    "val_heldout_shape",
    "test_extrapolation",
    "test_id",
    "val",
)
EVAL_SPLITS_BY_VERSION: dict[str, tuple[Split, ...]] = {
    "v1": EVAL_SPLITS,
    "v3": V3_EVAL_SPLITS,
    # protocol v4 replication: a fresh, separately frozen extrapolation suite
    "v4x": ("test_extrapolation",),
}


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class VerifierSpec(_Strict):
    kind: Literal["expression", "linear_equation"]
    text: str
    variable: Literal["x"] | None = None


class ProblemRecord(_Strict):
    id: str
    family: str
    template_id: str
    difficulty: int = Field(ge=1, le=5)
    prompt: str
    solution_steps: list[str] = Field(min_length=1)
    answer_type: Literal["integer", "rational"]
    final_answer: str
    verifier_spec: VerifierSpec
    generator_version: str
    seed: int
    split: Split
    number_range: Literal["in_distribution", "extrapolation"]
    params: dict[str, int]
    semantic_fingerprint: str


class CountsConfig(_Strict):
    train: PositiveInt
    val: PositiveInt
    test_id: PositiveInt
    test_extrapolation: PositiveInt
    test_heldout_template: PositiveInt  # per held-out template


class CorpusScanConfig(_Strict):
    enabled: bool = True
    sources_config: str = "configs/data_sources.yaml"
    raw_dir: str = "data/raw"


class DataBuildConfig(_Strict):
    dataset: Literal["math"] = "math"
    version: str = "v1"
    root_seed: int
    counts: CountsConfig
    count_overrides: dict[str, dict[Split, PositiveInt]] = Field(default_factory=dict)
    overgenerate: float = Field(default=1.5, ge=1.0, le=5.0)
    max_attempts_factor: PositiveInt = 60
    corpus_scan: CorpusScanConfig = CorpusScanConfig()
    out_dir: str = "data/generated/math_v1"
    manifest_dir: str = "data/manifests"
