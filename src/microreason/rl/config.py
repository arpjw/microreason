"""Strict configs for the RL-stage branches. Like stage configs, each resolves with its model
config inlined and is hashed, so checkpoints are bound to the config that produced them."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, PositiveInt

from microreason.config import BudgetConfig, ModelConfig, _resolve_ref, load_model_config


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class GateRule(_Strict):
    """Reward-variance gate (spec §6 Stage 5), declared before any GRPO run."""

    prompts: PositiveInt = 512
    min_mixed_group_fraction: float = Field(default=0.20, ge=0, le=1)
    max_malformed_fraction: float = Field(default=0.10, ge=0, le=1)
    min_pass_at_1: float = Field(default=0.05, ge=0, le=1)
    max_pass_at_1: float = Field(default=0.95, ge=0, le=1)


class SamplingConfig(_Strict):
    group_size: PositiveInt = 4
    temperature: float = Field(default=0.8, gt=0)
    max_new_tokens: PositiveInt = 128
    generation_batch_size: PositiveInt = 128


class GRPOConfig(_Strict):
    stage: Literal["grpo"] = "grpo"
    model: str
    tokenizer: str = "artifacts/tokenizer/v1"
    parent: str | None = None
    pool_manifest: str = "data/manifests/math_train_stream_rl1.json"
    seed: int = 45
    precision: Literal["bf16", "fp32"] = "bf16"
    sampling: SamplingConfig = SamplingConfig()
    gate: GateRule = GateRule()
    prompts_per_step: PositiveInt = 32
    steps: PositiveInt = 300
    minibatches_per_step: PositiveInt = 2
    clip_eps: float = Field(default=0.2, gt=0, lt=1)
    kl_beta: float = Field(default=0.04, ge=0)
    lr: float = Field(default=5e-6, gt=0)
    warmup_steps: int = Field(default=10, ge=0)
    weight_decay: float = Field(default=0.0, ge=0)
    betas: tuple[float, float] = (0.9, 0.95)
    grad_clip_norm: float = Field(default=1.0, gt=0)
    adv_eps: float = Field(default=1e-6, gt=0)
    log_every_steps: PositiveInt = 5
    checkpoint_every_steps: PositiveInt = 100
    budget: BudgetConfig = BudgetConfig()


class DPOConfig(_Strict):
    stage: Literal["dpo"] = "dpo"
    model: str
    tokenizer: str = "artifacts/tokenizer/v1"
    parent: str | None = None
    pool_manifest: str = "data/manifests/math_train_stream_rl1.json"
    seed: int = 46
    precision: Literal["bf16", "fp32"] = "bf16"
    sampling: SamplingConfig = SamplingConfig(generation_batch_size=256)
    max_pair_prompts: PositiveInt = 16_000
    target_pairs: PositiveInt = 6_000
    heldout_fraction: float = Field(default=0.10, gt=0, lt=1)
    beta: float = Field(default=0.1, gt=0)
    lr: float = Field(default=2e-6, gt=0)
    warmup_fraction: float = Field(default=0.1, ge=0, lt=1)
    batch_pairs: PositiveInt = 32
    epochs: PositiveInt = 1
    weight_decay: float = Field(default=0.0, ge=0)
    betas: tuple[float, float] = (0.9, 0.95)
    grad_clip_norm: float = Field(default=1.0, gt=0)
    log_every_steps: PositiveInt = 10
    budget: BudgetConfig = BudgetConfig()


class ResolvedRL(_Strict):
    schema_version: Literal[1] = 1
    stage: GRPOConfig | DPOConfig = Field(discriminator="stage")
    model: ModelConfig

    def config_hash(self) -> str:
        blob = json.dumps(self.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
        return "sha256:" + hashlib.sha256(blob.encode()).hexdigest()


def load_rl_config(path: str | Path, overrides: dict | None = None) -> ResolvedRL:
    path = Path(path)
    data = yaml.safe_load(path.read_text())
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected a YAML mapping")
    data.update({k: v for k, v in (overrides or {}).items() if v is not None})
    cls = {"grpo": GRPOConfig, "dpo": DPOConfig}.get(data.get("stage"))
    if cls is None:
        raise ValueError(f"{path}: stage must be grpo or dpo")
    stage = cls.model_validate(data)
    return ResolvedRL(stage=stage, model=load_model_config(_resolve_ref(stage.model, path)))
