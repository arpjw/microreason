"""Strict configs for the RL-stage branches. Like stage configs, each resolves with its model
config inlined and is hashed, so checkpoints are bound to the config that produced them."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, PositiveInt, model_validator

from microreason.config import BudgetConfig, ModelConfig, _resolve_ref, load_model_config


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class GateRule(_Strict):
    """Reward-variance gate (spec §6 Stage 5), declared before any GRPO run."""

    prompts: PositiveInt = 512
    # v1-v3 rule. None disables it (protocol v4 replaces it with `min_filter_yield`).
    min_mixed_group_fraction: float | None = Field(default=0.20, ge=0, le=1)
    max_malformed_fraction: float = Field(default=0.10, ge=0, le=1)
    min_pass_at_1: float = Field(default=0.05, ge=0, le=1)
    max_pass_at_1: float = Field(default=0.95, ge=0, le=1)
    # Protocol v4: the fraction of gate prompts whose group passes the GRPO config's training
    # filter (`filter_min_correct` <= correct <= `filter_max_correct`, rewards not all equal)
    # must be at least this, i.e. a bound on sampling cost per trained group. None disables it.
    min_filter_yield: float | None = Field(default=None, gt=0, le=1)


class SamplingConfig(_Strict):
    group_size: PositiveInt = 4
    temperature: float = Field(default=0.8, gt=0)
    max_new_tokens: PositiveInt = 128
    generation_batch_size: PositiveInt = 128
    # Rollouts only: incremental decoding with a KV cache (model.KVCache). Evaluation never
    # uses it. False reproduces the v1 sampler exactly.
    kv_cache: bool = False


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
    # Dynamic sampling (protocol rl-v2): sample prompts in chunks of `sample_chunk_prompts`
    # and keep only groups whose rewards mix successes and failures, until `prompts_per_step`
    # such groups are collected or `max_sample_rounds` chunks have been drawn. Dropped groups
    # carry zero advantage anyway; they are recorded but not trained on. Off reproduces v1.
    dynamic_sampling: bool = False
    sample_chunk_prompts: PositiveInt = 64
    max_sample_rounds: PositiveInt = 8
    minibatches_per_step: PositiveInt = 2
    clip_eps: float = Field(default=0.2, gt=0, lt=1)
    kl_beta: float = Field(default=0.04, ge=0)
    # ---- protocol v4 options (defaults reproduce the v1-v3 objective exactly)
    # "grpo": clipped policy gradient with a group baseline. "rft": on-policy rejection-sampling
    # fine-tuning (expert iteration / RAFT): advantage = reward with no baseline, i.e. the
    # likelihood of every correct sample is raised and nothing else; same sampler and filter.
    objective: Literal["grpo", "rft"] = "grpo"
    # group_std: (r - group mean) / group std (v1-v3). group_mean: r - group mean (Dr. GRPO).
    # batch_std: (r - group mean) / std of the rewards of all trained completions in the step.
    advantage: Literal["group_std", "group_mean", "batch_std"] = "group_std"
    # sequence_mean: tokens averaged per completion, then over completions (v1-v3).
    # token_constant: sum over every completion token / (completions x max_new_tokens)
    # (Dr. GRPO's constant normalizer: no length bias).
    loss_aggregation: Literal["sequence_mean", "token_constant"] = "sequence_mean"
    clip_eps_high: float | None = Field(default=None, gt=0, lt=1)  # None: symmetric clip_eps
    # Training filter under dynamic sampling: keep a group iff filter_min_correct <= correct
    # samples <= filter_max_correct and its rewards are not all equal. None means 1 and
    # group_size - 1 (the v2/v3 "mixed group" rule).
    filter_min_correct: int | None = Field(default=None, ge=0)
    filter_max_correct: int | None = Field(default=None, ge=0)
    mask_truncated: bool = False  # drop truncated completions from the loss (DAPO)
    # Bounded process reward (protocol v4 secondary arm): reward = [exact answer] + w x the
    # fraction of gold solution steps reproduced exactly as a prefix. w < 1, so every correct
    # answer outscores every wrong one. 0 is the binary primary reward.
    step_reward_weight: float = Field(default=0.0, ge=0, lt=1)
    update_micro_batch: PositiveInt = 256  # gradient accumulation within one optimizer update
    lr: float = Field(default=5e-6, gt=0)
    warmup_steps: int = Field(default=10, ge=0)
    weight_decay: float = Field(default=0.0, ge=0)
    betas: tuple[float, float] = (0.9, 0.95)
    grad_clip_norm: float = Field(default=1.0, gt=0)
    adv_eps: float = Field(default=1e-6, gt=0)
    log_every_steps: PositiveInt = 5
    checkpoint_every_steps: PositiveInt = 100
    budget: BudgetConfig = BudgetConfig()

    def filter_bounds(self) -> tuple[int, int]:
        k = self.sampling.group_size
        lo = 1 if self.filter_min_correct is None else self.filter_min_correct
        hi = k - 1 if self.filter_max_correct is None else self.filter_max_correct
        return lo, hi

    @model_validator(mode="after")
    def _check_v4_options(self) -> GRPOConfig:
        lo, hi = self.filter_bounds()
        if not 0 <= lo <= hi <= self.sampling.group_size:
            raise ValueError(f"filter bounds {lo}..{hi} invalid for group size "
                             f"{self.sampling.group_size}")  # fmt: skip
        if self.clip_eps_high is not None and self.clip_eps_high < self.clip_eps:
            raise ValueError("clip_eps_high must be >= clip_eps")
        return self


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
