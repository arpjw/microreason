"""Typed configuration schema for MicroReason.

Every config is validated strictly (unknown keys are rejected) before any work starts. A resolved
config is the stage config with its model config inlined and all defaults filled in; its canonical
JSON form is hashed so checkpoints can be bound to the exact config that produced them (spec §2,
invariant 3).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal, Self

import yaml
from pydantic import BaseModel, ConfigDict, Field, PositiveInt, field_validator, model_validator


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ModelConfig(_Strict):
    """Decoder-only Transformer shape (spec §3)."""

    name: str
    n_layers: PositiveInt
    d_model: PositiveInt
    n_heads: PositiveInt
    n_kv_heads: PositiveInt
    head_dim: PositiveInt
    mlp_hidden: PositiveInt
    vocab_size: PositiveInt
    max_seq_len: PositiveInt
    norm: Literal["rmsnorm"] = "rmsnorm"
    position: Literal["rope"] = "rope"
    rope_theta: float = Field(default=10000.0, gt=0)
    mlp: Literal["swiglu"] = "swiglu"
    tie_embeddings: bool = True
    bias: bool = False
    dropout: float = Field(default=0.0, ge=0.0, lt=1.0)

    @model_validator(mode="after")
    def _check_shape(self) -> Self:
        if self.n_heads * self.head_dim != self.d_model:
            raise ValueError(
                f"n_heads * head_dim ({self.n_heads} * {self.head_dim}) must equal "
                f"d_model ({self.d_model})"
            )
        if self.n_heads % self.n_kv_heads != 0:
            raise ValueError("n_heads must be a multiple of n_kv_heads")
        return self

    def parameter_count(self) -> int:
        """Analytic parameter count for the architecture as configured.

        This must later be reconciled against the instantiated model's actual count (spec §3).
        """
        d, m = self.d_model, self.mlp_hidden
        q_out = self.n_heads * self.head_dim
        kv_out = self.n_kv_heads * self.head_dim

        attn = d * q_out + 2 * d * kv_out + q_out * d
        mlp = 3 * d * m  # SwiGLU: gate, up, down
        norms = 2 * d  # RMSNorm before attention and before MLP
        if self.bias:
            attn += q_out + 2 * kv_out + d
            mlp += 2 * m + d
        block = attn + mlp + norms

        embed = self.vocab_size * d
        lm_head = 0 if self.tie_embeddings else self.vocab_size * d
        final_norm = d
        return embed + self.n_layers * block + final_norm + lm_head


class OptimizerConfig(_Strict):
    name: Literal["adamw"] = "adamw"
    peak_lr: float = Field(gt=0)
    weight_decay: float = Field(default=0.1, ge=0)
    betas: tuple[float, float] = (0.9, 0.95)
    eps: float = Field(default=1e-8, gt=0)
    grad_clip_norm: float = Field(default=1.0, gt=0)


class SchedulerConfig(_Strict):
    name: Literal["cosine", "constant"] = "cosine"
    warmup_fraction: float = Field(default=0.02, ge=0, lt=1)
    min_lr_fraction: float = Field(default=0.1, ge=0, le=1)


class BudgetConfig(_Strict):
    """Spending guard (spec §8). The default allows no paid compute at all."""

    paid_compute_allowed: bool = False
    max_spend_usd: float = Field(default=0.0, ge=0)
    provider: str | None = None
    gpu_type: str | None = None
    max_runtime_hours: float | None = Field(default=None, gt=0)
    hourly_rate_usd: float | None = Field(default=None, gt=0)
    shutdown_rule: str | None = None
    approved_by: str | None = None
    approved_at: str | None = None
    # Time held back from the ceilings to write a final checkpoint and copy results off.
    reserve_minutes: float = Field(default=10.0, ge=0)

    @field_validator("approved_at", mode="before")
    @classmethod
    def _date_to_str(cls, v: object) -> object:
        # YAML parses an unquoted 2026-11-07 as a date; store approvals as ISO strings.
        import datetime as _dt

        return v.isoformat() if isinstance(v, _dt.date) else v

    @model_validator(mode="after")
    def _check_authorization(self) -> Self:
        if not self.paid_compute_allowed:
            if self.max_spend_usd != 0:
                raise ValueError("max_spend_usd must be 0 when paid_compute_allowed is false")
            return self
        required = {
            "provider": self.provider,
            "gpu_type": self.gpu_type,
            "max_runtime_hours": self.max_runtime_hours,
            "hourly_rate_usd": self.hourly_rate_usd,
            "shutdown_rule": self.shutdown_rule,
            "approved_by": self.approved_by,
            "approved_at": self.approved_at,
        }
        missing = [k for k, v in required.items() if v in (None, "")]
        if missing:
            raise ValueError(f"paid compute requires these budget fields: {', '.join(missing)}")
        if self.max_spend_usd <= 0:
            raise ValueError("paid compute requires a positive max_spend_usd")
        placeholders = {"PENDING", "TBD", "TODO", "CHANGEME"}
        unapproved = [
            k for k in ("approved_by", "approved_at")
            if str(getattr(self, k)).strip().upper() in placeholders
        ]  # fmt: skip
        if unapproved:
            raise ValueError(
                f"paid compute is not authorized: {', '.join(unapproved)} still a placeholder"
            )
        return self

    def ceiling_hours(self) -> float:
        """The tighter of the runtime limit and the dollar cap at the listed rate."""
        if not self.paid_compute_allowed:
            return 0.0
        return min(self.max_runtime_hours, self.max_spend_usd / self.hourly_rate_usd)


Stage = Literal["smoke", "pretrain", "midtrain", "sft", "extra_sft", "dpo", "grpo"]


class MixtureItem(_Strict):
    """A pre-tokenized dataset directory and its share of processed tokens."""

    dataset: str
    weight: float = Field(gt=0)


class ValidationItem(_Strict):
    dataset: str
    sequences: PositiveInt = 64


class StageConfig(_Strict):
    """One training stage (spec §6). `model` is a path to a ModelConfig YAML file."""

    stage: Stage
    model: str
    tokenizer: str | None = None
    data_manifest: str | None = None
    mixture: list[MixtureItem] = Field(default_factory=list)
    validation: list[ValidationItem] = Field(default_factory=list)
    parent: str | None = None
    seed: int = 42
    precision: Literal["bf16", "fp16", "fp32"] = "bf16"
    sequence_length: PositiveInt
    effective_batch_tokens: PositiveInt
    micro_batch_size: PositiveInt | None = None  # sequences per forward pass; None = whole batch
    max_train_tokens: PositiveInt | None = None
    max_steps: PositiveInt | None = None
    max_examples: PositiveInt | None = None
    log_every_steps: PositiveInt = 10
    eval_every_steps: PositiveInt = 250
    checkpoint_every_steps: PositiveInt = 500
    optimizer: OptimizerConfig
    scheduler: SchedulerConfig = SchedulerConfig()
    budget: BudgetConfig = BudgetConfig()

    @model_validator(mode="after")
    def _check_stage(self) -> Self:
        if self.effective_batch_tokens % self.sequence_length != 0:
            raise ValueError("effective_batch_tokens must be a multiple of sequence_length")
        if self.max_train_tokens is None and self.max_steps is None and self.max_examples is None:
            raise ValueError("set at least one of max_train_tokens, max_steps, max_examples")
        if self.stage in ("midtrain", "sft", "extra_sft", "dpo", "grpo") and self.parent is None:
            raise ValueError(f"stage '{self.stage}' requires a parent checkpoint")
        if self.micro_batch_size is not None and self.sequences_per_step % self.micro_batch_size:
            raise ValueError("sequences per step must be a multiple of micro_batch_size")
        return self

    @property
    def sequences_per_step(self) -> int:
        return self.effective_batch_tokens // self.sequence_length

    @property
    def grad_accum_steps(self) -> int:
        return self.sequences_per_step // (self.micro_batch_size or self.sequences_per_step)

    def total_steps(self) -> int:
        if self.max_steps is not None:
            return self.max_steps
        if self.max_train_tokens is not None:
            return -(-self.max_train_tokens // self.effective_batch_tokens)  # ceil
        raise ValueError("example-limited stages compute steps from their dataset")


class ResolvedConfig(_Strict):
    """A stage config with its model config inlined: the unit that gets hashed."""

    schema_version: Literal[1] = 1
    stage: StageConfig
    model: ModelConfig

    @model_validator(mode="after")
    def _check_fit(self) -> Self:
        if self.stage.sequence_length > self.model.max_seq_len:
            raise ValueError(
                f"sequence_length {self.stage.sequence_length} exceeds model max_seq_len "
                f"{self.model.max_seq_len}"
            )
        return self

    def canonical_json(self) -> str:
        return json.dumps(self.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))

    def config_hash(self) -> str:
        return "sha256:" + hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()


def _read_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected a YAML mapping at top level")
    return data


def _resolve_ref(ref: str, config_path: Path) -> Path:
    """Resolve a referenced path relative to the working directory, else the config's folder."""
    candidate = Path(ref)
    if candidate.is_absolute() or candidate.exists():
        return candidate
    return config_path.parent / candidate


def load_model_config(path: str | Path) -> ModelConfig:
    path = Path(path)
    return ModelConfig.model_validate(_read_yaml(path))


def load_stage_config(path: str | Path) -> ResolvedConfig:
    path = Path(path)
    stage = StageConfig.model_validate(_read_yaml(path))
    model = load_model_config(_resolve_ref(stage.model, path))
    return ResolvedConfig(stage=stage, model=model)
