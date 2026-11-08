"""The common training loop shared by every causal-LM stage (spec §6, Stage 0).

Features: AdamW with a declared weight-decay policy, linear warmup plus cosine decay, gradient
accumulation weighted by target tokens, global-norm clipping, periodic evaluation, structured
metrics, atomic checkpoints and exact resume.
"""

from __future__ import annotations

import contextlib
import json
import math
import platform
import random
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import torch

from microreason.checkpoint import load_checkpoint, save_checkpoint
from microreason.config import ResolvedConfig, SchedulerConfig
from microreason.logging_utils import get_logger
from microreason.train.data import PackedStream

log = get_logger("microreason.train")

WEIGHT_DECAY_POLICY = "decay all Linear weights; no decay on RMSNorm weights or the embedding"


class NonFiniteError(RuntimeError):
    pass


def unpack_batch(b) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """A batch is either a token tensor (packed LM: every position after the first is a target)
    or a dict with input_ids, labels and attention_mask (SFT: padded, prompt/padding masked)."""
    if isinstance(b, torch.Tensor):
        return b, b, None
    return b["input_ids"], b["labels"], b.get("attention_mask")


def count_targets(labels: torch.Tensor) -> int:
    from microreason.prompts import IGNORE_INDEX

    return int((labels[:, 1:] != IGNORE_INDEX).sum())


def count_tokens(ids: torch.Tensor, mask: torch.Tensor | None) -> int:
    return int(mask.sum()) if mask is not None else ids.numel()


def lr_at(step: int, total_steps: int, peak: float, sched: SchedulerConfig) -> float:
    """Learning rate for optimizer update number `step` (0-based)."""
    warm = math.ceil(sched.warmup_fraction * total_steps) if sched.warmup_fraction > 0 else 0
    if step < warm:
        return peak * (step + 1) / warm
    if sched.name == "constant":
        return peak
    lo = peak * sched.min_lr_fraction
    progress = (step - warm) / max(1, total_steps - warm - 1)
    return lo + (peak - lo) * 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))


def param_groups(model: torch.nn.Module, weight_decay: float) -> list[dict]:
    decay, no_decay = [], []
    embed = model.get_input_embeddings().weight
    for p in model.parameters():
        (no_decay if p.ndim < 2 or p is embed else decay).append(p)
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


def _autocast(device: torch.device, precision: str):
    if precision == "fp32":
        return contextlib.nullcontext()
    if precision == "bf16" and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    if precision == "bf16" and device.type == "mps":
        # Measured 2026-11-08: a full 43M bf16 step on MPS did not finish in 10 minutes.
        raise ValueError("bf16 autocast on MPS is impractically slow; use precision fp32 locally")
    raise ValueError(f"precision {precision} is not supported on {device.type}")


def _code_revision() -> dict:
    import subprocess

    try:
        rev = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"], capture_output=True, text=True, check=True
            ).stdout.strip()
        )
        return {"git_commit": rev, "dirty": dirty}
    except (OSError, subprocess.CalledProcessError):
        rev_file = Path("REVISION")
        if rev_file.exists():  # written by scripts/pod_push.sh for git-less copies
            return {"git_commit": rev_file.read_text().strip(), "dirty": False,
                    "source": "REVISION file"}  # fmt: skip
        return {"git_commit": None, "dirty": None}


@dataclass
class TrainResult:
    run_dir: Path
    final_checkpoint: Path | None
    steps: int
    tokens: int
    final_eval: dict
    stop_reason: str | None = None


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def peak_memory_gib(device: torch.device) -> float | None:
    if device.type == "cuda":
        return round(torch.cuda.max_memory_allocated(device) / 2**30, 3)
    if device.type == "mps":
        return round(torch.mps.driver_allocated_memory() / 2**30, 3)
    return None


class Trainer:
    def __init__(
        self,
        resolved: ResolvedConfig,
        run_dir: str | Path,
        stream: PackedStream,
        eval_sets: dict[str, list[torch.Tensor]],
        device: str | torch.device = "cpu",
        provenance: dict | None = None,
        model=None,
        budget_guard=None,
    ):
        self.cfg = resolved
        self.stage = resolved.stage
        if self.stage.budget.paid_compute_allowed and budget_guard is None:
            raise ValueError("paid compute runs need a budget guard")
        self.guard = budget_guard
        self.run_dir = Path(run_dir)
        self.device = torch.device(device)
        if self.device.type == "cuda":
            torch.set_float32_matmul_precision("high")
        self.stream = stream
        self.eval_sets = eval_sets
        self.total_steps = self.stage.total_steps()
        self.micro = self.stage.micro_batch_size or self.stage.sequences_per_step
        self.accum = self.stage.grad_accum_steps

        torch.manual_seed(self.stage.seed)
        random.seed(self.stage.seed)
        if model is None:
            from microreason.model import build_model

            model = build_model(resolved.model)
        self.model = model.to(self.device)
        opt = self.stage.optimizer
        self.optimizer = torch.optim.AdamW(
            param_groups(self.model, opt.weight_decay),
            lr=opt.peak_lr,
            betas=opt.betas,
            eps=opt.eps,
            foreach=False,
        )
        self.step = 0
        self.tokens = 0
        self.examples = 0
        self.provenance = {
            "stage": self.stage.stage,
            "config_hash": resolved.config_hash(),
            "parent": self.stage.parent,
            "code_revision": _code_revision(),
            "weight_decay_policy": WEIGHT_DECAY_POLICY,
            "packing_policy": "bos/eos-separated documents concatenated; attention crosses docs",
            **(provenance or {}),
        }

    # ------------------------------------------------------------------ io

    def _event(self, kind: str, **data) -> None:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        row = {"event": kind, "step": self.step, "tokens": self.tokens, **data}
        with (self.run_dir / "metrics.jsonl").open("a") as f:
            f.write(json.dumps(row, sort_keys=True) + "\n")

    def _write_manifest(self, status: str, **extra) -> None:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        path = self.run_dir / "run_manifest.json"
        prev = json.loads(path.read_text()) if path.exists() else {}
        manifest = {
            **prev,
            "status": status,
            "resolved_config": self.cfg.model_dump(mode="json"),
            "provenance": self.provenance,
            "device": str(self.device),
            "software": {
                "python": platform.python_version(),
                "torch": torch.__version__,
                "platform": platform.platform(),
            },
            "total_steps": self.total_steps,
            "steps_completed": self.step,
            "tokens_processed": self.tokens,
            "examples_processed": self.examples,
            "estimated_cost_usd": 0.0,
            "updated_at": datetime.now(UTC).isoformat(),
            **extra,
        }
        manifest.setdefault("started_at", manifest["updated_at"])
        path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    def _state(self) -> dict:
        return {
            "step": self.step,
            "tokens": self.tokens,
            "examples": self.examples,
            "stream": self.stream.state_dict(),
            "torch_rng": torch.get_rng_state(),
            "python_rng": random.getstate(),
        }

    def save(self) -> Path:
        path = save_checkpoint(
            self.run_dir,
            self.step,
            self.model,
            self.optimizer,
            self._state(),
            {**self.provenance, "tokens": self.tokens, "stream_digest": self.stream.digest},
        )
        self._event("checkpoint", path=str(path))
        return path

    def resume(self, checkpoint: str | Path) -> None:
        model, state, manifest = load_checkpoint(checkpoint, "cpu")
        if manifest["config_hash"] != self.provenance["config_hash"]:
            raise ValueError("checkpoint was written under a different config; refusing to resume")
        self.model.load_state_dict(model.state_dict())
        self.optimizer.load_state_dict(state["optimizer"])
        self.stream.load_state_dict(state["stream"])
        self.step = state["step"]
        self.tokens = state["tokens"]
        self.examples = state.get("examples", 0)
        torch.set_rng_state(state["torch_rng"])
        py = state["python_rng"]
        random.setstate((py[0], tuple(py[1]), py[2]))
        self._event("resume", checkpoint=str(checkpoint))
        log.info("resumed from %s at step %d (%d tokens)", checkpoint, self.step, self.tokens)

    # ------------------------------------------------------------------ eval

    @torch.no_grad()
    def evaluate(self) -> dict:
        self.model.eval()
        out = {}
        for name, batches in self.eval_sets.items():
            total, n = 0.0, 0
            for b in batches:
                ids, labels, mask = (None if t is None else t.to(self.device)
                                     for t in unpack_batch(b))  # fmt: skip
                with _autocast(self.device, self.stage.precision):
                    loss = self.model(ids, attention_mask=mask, labels=labels).loss.float().item()
                k = count_targets(labels)
                total += loss * k
                n += k
            out[name] = total / n
        self.model.train()
        return out

    # ------------------------------------------------------------------ loop

    def accumulate_gradients(self) -> tuple[float, int]:
        """Forward/backward over one optimizer step's micro-batches.

        Each micro-batch's mean loss is weighted by its share of the step's target tokens, so
        the accumulated gradient equals the gradient of the full batch's mean loss.
        """
        batches = [unpack_batch(self.stream.next_batch(self.micro)) for _ in range(self.accum)]
        n_targets = sum(count_targets(labels) for _, labels, _ in batches)
        loss_sum = 0.0
        n_tokens = 0
        for ids, labels, mask in batches:
            ids, labels = ids.to(self.device), labels.to(self.device)
            mask = None if mask is None else mask.to(self.device)
            with _autocast(self.device, self.stage.precision):
                loss = self.model(ids, attention_mask=mask, labels=labels).loss
            k = count_targets(labels)
            (loss * (k / n_targets)).backward()
            loss_sum += loss.detach().float().item() * k
            n_tokens += count_tokens(ids, mask)
        self.examples += sum(ids.shape[0] for ids, _, _ in batches)
        return loss_sum / n_targets, n_tokens

    def train_step(self) -> dict:
        lr = lr_at(self.step, self.total_steps, self.stage.optimizer.peak_lr, self.stage.scheduler)
        for g in self.optimizer.param_groups:
            g["lr"] = lr
        train_loss, n_tokens = self.accumulate_gradients()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), self.stage.optimizer.grad_clip_norm
        ).item()
        if not (math.isfinite(train_loss) and math.isfinite(grad_norm)):
            raise NonFiniteError(f"step {self.step}: loss={train_loss} grad_norm={grad_norm}")
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        self.step += 1
        self.tokens += n_tokens
        return {"loss": train_loss, "grad_norm": grad_norm, "lr": lr}

    def train(self, resume_from: str | Path | None = None) -> TrainResult:
        if resume_from is not None:
            self.resume(resume_from)
        self.model.train()
        self._write_manifest("running")
        log.info(
            "stage=%s steps=%d micro_batch=%d accum=%d seq=%d device=%s",
            self.stage.stage, self.total_steps, self.micro, self.accum,
            self.stage.sequence_length, self.device,
        )  # fmt: skip
        last_ckpt, last_eval = None, {}
        t0, tok0 = time.time(), self.tokens
        step_times: list[float] = []
        steps_this_run = 0
        stop_reason = None
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
        wall_limit = self.stage.max_wall_seconds
        loop_start = time.time()
        wall_stop = False
        try:
            while self.step < self.total_steps:
                if wall_limit is not None and time.time() - loop_start >= wall_limit:
                    wall_stop = True
                    log.info("wall-time limit %.0fs reached at step %d", wall_limit, self.step)
                    break
                if self.guard is not None:
                    est = sorted(step_times[-5:])[len(step_times[-5:]) // 2] if step_times else None
                    remaining = self.total_steps - self.step
                    if wall_limit is not None and est:
                        left = max(0.0, wall_limit - (time.time() - loop_start))
                        remaining = min(remaining, math.ceil(left / est))
                    stop_reason = self.guard.check(est, steps_this_run, remaining)
                    if stop_reason:
                        log.warning("budget guard stopped the run: %s", stop_reason)
                        break
                s0 = time.time()
                m = self.train_step()
                _sync(self.device)
                step_times.append(time.time() - s0)
                steps_this_run += 1
                done = self.step == self.total_steps
                if self.step % self.stage.log_every_steps == 0 or done:
                    dt = time.time() - t0
                    tps = (self.tokens - tok0) / dt if dt > 0 else 0.0
                    self._event(
                        "train",
                        tokens_per_second=round(tps, 1),
                        step_seconds=round(step_times[-1], 4),
                        peak_memory_gib=peak_memory_gib(self.device),
                        **m,
                    )
                    log.info("step %d/%d loss %.4f gnorm %.3f lr %.2e %.0f tok/s",
                             self.step, self.total_steps, m["loss"], m["grad_norm"], m["lr"],
                             tps)  # fmt: skip
                    t0, tok0 = time.time(), self.tokens
                if self.step % self.stage.eval_every_steps == 0 or done:
                    last_eval = self.evaluate()
                    self._event("eval", **{f"val_loss/{k}": v for k, v in last_eval.items()})
                    log.info("eval step %d %s", self.step,
                             {k: round(v, 4) for k, v in last_eval.items()})  # fmt: skip
                if self.step % self.stage.checkpoint_every_steps == 0 or done:
                    last_ckpt = self.save()
            if (stop_reason or wall_stop) and self.step > 0:
                from microreason.checkpoint import checkpoint_dir

                if not checkpoint_dir(self.run_dir, self.step).exists():
                    last_ckpt = self.save()
                else:
                    last_ckpt = checkpoint_dir(self.run_dir, self.step)
        except BaseException as e:
            self._write_manifest("failed", error=f"{type(e).__name__}: {e}")
            raise
        timing = {
            "median_step_seconds": round(sorted(step_times)[len(step_times) // 2], 4)
            if step_times
            else None,
            "peak_memory_gib": peak_memory_gib(self.device),
            "train_wall_seconds": round(time.time() - loop_start, 2),
            "stopped_at_wall_limit": wall_stop,
        }
        if wall_stop:
            last_eval = self.evaluate()
            self._event("eval", **{f"val_loss/{k}": v for k, v in last_eval.items()})
        if self.guard is not None:
            timing["budget_ledger"] = self.guard.write(
                timing["median_step_seconds"], self.total_steps - self.step, stop_reason
            )
        self._write_manifest(
            stop_reason or "completed",
            final_eval=last_eval,
            mixture=self.stream.mixture_report(),
            final_checkpoint=str(last_ckpt),
            timing=timing,
            estimated_cost_usd=round(self.guard.accrued_usd(), 4) if self.guard else 0.0,
        )
        return TrainResult(self.run_dir, last_ckpt, self.step, self.tokens, last_eval, stop_reason)
