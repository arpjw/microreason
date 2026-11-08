"""On-policy GRPO from v2-sft with the binary exact-answer reward (spec §6 Stage 5).

Implementation choices (stated as the spec requires):

- **Rollouts.** Each step takes the next `prompts_per_step` prompts of the shuffled `grpo`
  partition (no prompt repeats) and samples `group_size` completions per prompt from the current
  policy (temperature 0.8, at most 128 new tokens, show-work format).
- **Reward and advantage.** Binary primary reward (`rl.rollout`). Within each group,
  A_i = (r_i - mean) / (std + eps) with the population std. A group with zero reward variance
  gets A = 0 for every member: no relative signal is invented. Those groups still contribute
  the KL term.
- **Old-policy ratios.** After sampling, the generating weights recompute every completion
  token's log probability by teacher forcing (`old_lp`). The step's completions are shuffled
  into `minibatches_per_step` minibatches, each one optimizer update. The ratio
  exp(new_lp - old_lp) is exactly 1 for the first minibatch and reflects at most
  `minibatches_per_step - 1` updates of staleness after that. **Staleness rule:** a rollout is
  used only within the step that generated it; groups from different policy versions are never
  mixed.
- **Clipping.** PPO-style per-token surrogate min(r A, clip(r, 1 - eps, 1 + eps) A),
  eps = 0.2.
- **KL.** To the frozen v2-sft reference, with the k3 estimator per token,
  exp(ref - new) - (ref - new) - 1, weighted by `kl_beta`, added to the loss (not to the reward).
- **Temperature.** Log probabilities for the ratio and the KL use logits divided by the sampling
  temperature, i.e. the distribution the samples were drawn from.
- **Normalization.** Per-token terms are averaged over each completion's own tokens (sequence
  mean), then averaged over the minibatch's completions. Prompt and padding tokens are masked.
- **Truncation.** A completion that hits the token limit without [eos] is kept in the loss and
  scores 0 (the evaluation also counts it as a failure).
- **Schedule.** AdamW, linear warmup over `warmup_steps` rollout steps, then constant LR.

Every completion is appended to an immutable rollout record (`rollouts.jsonl`): prompt id,
policy version, sampled token ids, old log probabilities, reward, status and the generation
settings. Its sha256 goes into the run manifest.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import random
import time
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path

import torch

from microreason.logging_utils import get_logger
from microreason.rl.config import GRPOConfig, ResolvedRL
from microreason.rl.rollout import (
    MALFORMED,
    autocast,
    build_sequences,
    sample_groups,
    token_logprobs,
)

log = get_logger("microreason.rl.grpo")


def group_advantages(rewards: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """rewards: (groups, group_size) -> standardized advantages; zero for zero-variance groups."""
    mean = rewards.mean(dim=1, keepdim=True)
    std = rewards.std(dim=1, unbiased=False, keepdim=True)
    adv = (rewards - mean) / (std + eps)
    return torch.where(std > 0, adv, torch.zeros_like(adv))


def grpo_loss(
    new_lp: torch.Tensor,
    old_lp: torch.Tensor,
    ref_lp: torch.Tensor,
    mask: torch.Tensor,
    adv: torch.Tensor,
    clip_eps: float,
    kl_beta: float,
) -> tuple[torch.Tensor, dict]:
    """All log-prob tensors (B, T) fp32, mask (B, T) bool, adv (B,). Returns (loss, stats)."""
    m = mask.float()
    n_tok = m.sum(dim=1).clamp(min=1.0)
    log_ratio = (new_lp - old_lp) * m
    ratio = torch.exp(log_ratio)
    a = adv[:, None]
    surr = torch.minimum(ratio * a, torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * a)
    d = (ref_lp - new_lp) * m
    kl = torch.exp(d) - d - 1
    per_tok = -(surr - kl_beta * kl) * m
    loss = (per_tok.sum(dim=1) / n_tok).mean()
    with torch.no_grad():
        clipped = ((ratio - 1).abs() > clip_eps) & mask
        stats = {
            "kl": float(((kl * m).sum(dim=1) / n_tok).mean()),
            "clip_fraction": float(clipped.sum() / m.sum().clamp(min=1.0)),
            "ratio_mean": float((ratio * m).sum() / m.sum().clamp(min=1.0)),
        }
    return loss, stats


def _lr_at(step: int, cfg: GRPOConfig) -> float:
    if cfg.warmup_steps and step < cfg.warmup_steps:
        return cfg.lr * (step + 1) / cfg.warmup_steps
    return cfg.lr


class GRPOTrainer:
    def __init__(self, resolved: ResolvedRL, run_dir: str | Path, device: str, guard=None,
                 pool: list[dict] | None = None):  # fmt: skip
        from microreason.checkpoint import MANIFEST, load_checkpoint
        from microreason.rl.pool import load_partition
        from microreason.tokenizer import MicroTokenizer
        from microreason.train.common import _code_revision, param_groups

        cfg = resolved.stage
        if not isinstance(cfg, GRPOConfig):
            raise ValueError("GRPOTrainer needs a grpo config")
        if cfg.parent is None:
            raise ValueError("GRPO needs a parent checkpoint (v2-sft)")
        if cfg.budget.paid_compute_allowed and guard is None:
            raise ValueError("paid compute runs need a budget guard")
        self.cfg, self.resolved, self.guard = cfg, resolved, guard
        self.run_dir = Path(run_dir)
        if (self.run_dir / "rollouts.jsonl").exists():
            raise FileExistsError(f"{self.run_dir} already holds a GRPO run")
        self.device = torch.device(device)
        if self.device.type == "cuda":
            torch.set_float32_matmul_precision("high")
        torch.manual_seed(cfg.seed)
        self.tok = MicroTokenizer.load(cfg.tokenizer)
        self.policy, _, parent_manifest = load_checkpoint(cfg.parent, self.device)
        if parent_manifest.get("tokenizer_sha256") not in (None, self.tok.sha256):
            raise ValueError("parent was trained with a different tokenizer")
        self.ref = copy.deepcopy(self.policy).eval()
        for p in self.ref.parameters():
            p.requires_grad_(False)
        self.policy.train()
        self.optimizer = torch.optim.AdamW(param_groups(self.policy, cfg.weight_decay),
                                           lr=cfg.lr, betas=cfg.betas, foreach=False)  # fmt: skip
        if pool is None:
            pool, pool_manifest = load_partition(cfg.pool_manifest, "grpo")
            pool_sha = pool_manifest["sha256"]
        else:
            pool_sha = "in-memory (tests)"
        need = cfg.steps * cfg.prompts_per_step
        if len(pool) < need:
            raise ValueError(f"grpo partition has {len(pool)} prompts, need {need}")
        order = list(range(len(pool)))
        random.Random(f"{cfg.seed}|grpo-prompts").shuffle(order)
        self.prompts = [pool[i] for i in order[:need]]
        self.parent_id = (
            "sha256:" + hashlib.sha256((Path(cfg.parent) / MANIFEST).read_bytes()).hexdigest()
        )
        self.provenance = {
            "stage": "grpo",
            "config_hash": resolved.config_hash(),
            "parent": cfg.parent,
            "parent_checkpoint_id": self.parent_id,
            "parent_config_hash": parent_manifest.get("config_hash"),
            "tokenizer_sha256": self.tok.sha256,
            "code_revision": _code_revision(),
            "pool": {
                "manifest": cfg.pool_manifest,
                "sha256": pool_sha,
                "partition": "grpo",
                "prompts_used": need,
            },  # fmt: skip
            "objective": "GRPO, clipped ratio, k3 KL to frozen parent, sequence-mean tokens",
        }
        self.step = 0
        self.updates = 0
        self.gen = torch.Generator(device=self.device).manual_seed(cfg.seed)
        self.shuffle_rng = torch.Generator().manual_seed(cfg.seed + 1)

    # ------------------------------------------------------------------ io

    def _event(self, kind: str, **data) -> None:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        row = {"event": kind, "step": self.step, "updates": self.updates, **data}
        with (self.run_dir / "metrics.jsonl").open("a") as f:
            f.write(json.dumps(row, sort_keys=True) + "\n")

    def _write_manifest(self, status: str, **extra) -> None:
        import platform

        self.run_dir.mkdir(parents=True, exist_ok=True)
        path = self.run_dir / "run_manifest.json"
        prev = json.loads(path.read_text()) if path.exists() else {}
        m = {**prev, "status": status, "resolved_config": self.resolved.model_dump(mode="json"),
             "provenance": self.provenance, "device": str(self.device),
             "software": {"python": platform.python_version(), "torch": torch.__version__},
             "total_steps": self.cfg.steps, "steps_completed": self.step,
             "optimizer_updates": self.updates,
             "updated_at": datetime.now(UTC).isoformat(), **extra}  # fmt: skip
        m.setdefault("started_at", m["updated_at"])
        path.write_text(json.dumps(m, indent=2, sort_keys=True) + "\n")

    def save(self) -> Path:
        from microreason.checkpoint import save_checkpoint

        state = {"step": self.step, "updates": self.updates, "torch_rng": torch.get_rng_state()}
        return save_checkpoint(self.run_dir, self.step, self.policy, self.optimizer, state,
                               {**self.provenance, "tokens": 0})  # fmt: skip

    # ------------------------------------------------------------------ one step

    def train_step(self, rollout_file) -> dict:
        cfg, s = self.cfg, self.cfg.sampling
        lr = _lr_at(self.step, cfg)
        for g in self.optimizer.param_groups:
            g["lr"] = lr
        recs = self.prompts[
            self.step * cfg.prompts_per_step : (self.step + 1) * cfg.prompts_per_step
        ]
        t0 = time.time()
        groups = sample_groups(self.policy, self.tok, recs, group_size=s.group_size,
                               temperature=s.temperature, max_new_tokens=s.max_new_tokens,
                               generator=self.gen, batch_size=s.generation_batch_size,
                               precision=cfg.precision)  # fmt: skip
        t_roll = time.time() - t0
        rewards = torch.tensor([[c.reward for c in g] for g in groups])
        adv = group_advantages(rewards, cfg.adv_eps).flatten()
        from microreason.eval.generate import eval_prompt_ids
        from microreason.rl.rollout import ROLLOUT_FORMAT

        prompt_cache = [eval_prompt_ids(self.tok, r["prompt"], ROLLOUT_FORMAT) for r in recs]
        flat = [(gi, c) for gi, g in enumerate(groups) for c in g]
        # an empty completion still needs one target position; generate() always emits >= 1 id
        batch = build_sequences([prompt_cache[gi] for gi, _ in flat], [c.ids for _, c in flat],
                                self.tok.pad_id)  # fmt: skip
        batch = {k: v.to(self.device) for k, v in batch.items()}
        t1 = time.time()
        with torch.no_grad(), autocast(self.device, cfg.precision):
            old_lp = token_logprobs(self.policy, batch, s.temperature)
            ref_lp = token_logprobs(self.ref, batch, s.temperature)
        mask = batch["response_mask"]
        old_cpu, mask_cpu = old_lp.cpu(), mask.cpu()
        # immutable rollout record (one line per completion)
        for row, (gi, c) in enumerate(flat):
            n = int(mask_cpu[row].sum())
            lp = old_cpu[row][mask_cpu[row]].tolist()
            rollout_file.write(json.dumps({
                "prompt_id": recs[gi]["id"], "template_id": recs[gi]["template_id"],
                "step": self.step, "policy_updates_before": self.updates,
                "parent_checkpoint_id": self.parent_id, "token_ids": c.ids,
                "old_logprobs": [round(x, 5) for x in lp], "n_tokens": n,
                "reward": c.reward, "status": c.status, "reason": c.reason,
                "hit_eos": c.hit_eos, "advantage": round(float(adv[row]), 6),
                "generation": {"temperature": s.temperature, "max_new_tokens": s.max_new_tokens,
                               "group_size": s.group_size, "format": ROLLOUT_FORMAT},
            }, sort_keys=True) + "\n")  # fmt: skip
        adv = adv.to(self.device)
        perm = torch.randperm(len(flat), generator=self.shuffle_rng)
        stats = defaultdict(float)
        grad_norms = []
        for chunk in perm.chunk(cfg.minibatches_per_step):
            idx = chunk.to(self.device)
            sub = {k: v[idx] for k, v in batch.items()}
            with autocast(self.device, cfg.precision):
                new_lp = token_logprobs(self.policy, sub, s.temperature)
            loss, st = grpo_loss(new_lp, old_lp[idx], ref_lp[idx], sub["response_mask"],
                                 adv[idx], cfg.clip_eps, cfg.kl_beta)  # fmt: skip
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(self.policy.parameters(), cfg.grad_clip_norm)
            gn = float(gn)
            if not (math.isfinite(float(loss.detach())) and math.isfinite(gn)):
                raise RuntimeError(f"non-finite GRPO loss/grad at step {self.step}")
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)
            self.updates += 1
            grad_norms.append(gn)
            stats["loss"] += float(loss.detach()) / cfg.minibatches_per_step
            for k, v in st.items():
                stats[k] += v / cfg.minibatches_per_step
        t_upd = time.time() - t1
        self.step += 1
        succ = rewards.sum(dim=1)
        k = s.group_size
        comps = [c for g in groups for c in g]
        by_family: dict[str, list[float]] = defaultdict(list)
        for r, g in zip(recs, groups, strict=True):
            by_family[r["family"]] += [c.reward for c in g]
        return {
            "lr": lr,
            "mean_reward": float(rewards.mean()),
            "mixed_group_fraction": float(((succ > 0) & (succ < k)).float().mean()),
            "all_correct_fraction": float((succ == k).float().mean()),
            "all_wrong_fraction": float((succ == 0).float().mean()),
            "malformed_fraction": sum(c.status in MALFORMED for c in comps) / len(comps),
            "truncated_fraction": sum(c.status == "truncated" for c in comps) / len(comps),
            "verifier_errors": sum(c.status == "verifier_error" for c in comps),
            "mean_response_tokens": sum(len(c.ids) for c in comps) / len(comps),
            "grad_norm": max(grad_norms),
            "success_by_family": {f: sum(v) / len(v) for f, v in sorted(by_family.items())},
            "status_counts": dict(Counter(c.status for c in comps)),
            "rollout_seconds": round(t_roll, 3),
            "update_seconds": round(t_upd, 3),
            **{k2: round(v, 6) for k2, v in stats.items()},
        }

    # ------------------------------------------------------------------ loop

    def train(self) -> dict:
        from microreason.checkpoint import checkpoint_dir
        from microreason.train.common import peak_memory_gib

        cfg = self.cfg
        self._write_manifest("running")
        log.info("grpo steps=%d prompts/step=%d group=%d minibatches=%d device=%s", cfg.steps,
                 cfg.prompts_per_step, cfg.sampling.group_size, cfg.minibatches_per_step,
                 self.device)  # fmt: skip
        step_times: list[float] = []
        stop_reason = None
        last = None
        totals = Counter()
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
        loop_start = time.time()
        rollout_path = self.run_dir / "rollouts.jsonl"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        try:
            with rollout_path.open("x") as rf:
                while self.step < cfg.steps:
                    if self.guard is not None:
                        recent = sorted(step_times[-5:])
                        est = recent[len(recent) // 2] if recent else None
                        stop_reason = self.guard.check(est, len(step_times), cfg.steps - self.step)
                        if stop_reason:
                            log.warning("budget guard stopped GRPO: %s", stop_reason)
                            break
                    t0 = time.time()
                    m = self.train_step(rf)
                    step_times.append(time.time() - t0)
                    totals["completions"] += cfg.prompts_per_step * cfg.sampling.group_size
                    totals["rollout_seconds"] += m["rollout_seconds"]
                    totals["update_seconds"] += m["update_seconds"]
                    self._event("train", step_seconds=round(step_times[-1], 3),
                                peak_memory_gib=peak_memory_gib(self.device), **m)  # fmt: skip
                    if self.step % cfg.log_every_steps == 0 or self.step == cfg.steps:
                        log.info("step %d/%d reward %.3f mixed %.2f kl %.4f clip %.3f len %.1f "
                                 "%.1fs", self.step, cfg.steps, m["mean_reward"],
                                 m["mixed_group_fraction"], m["kl"], m["clip_fraction"],
                                 m["mean_response_tokens"], step_times[-1])  # fmt: skip
                        rf.flush()
                    if self.step % cfg.checkpoint_every_steps == 0 or self.step == cfg.steps:
                        last = self.save()
            if (
                stop_reason
                and self.step > 0
                and not checkpoint_dir(self.run_dir, self.step).exists()
            ):
                last = self.save()
        except BaseException as e:
            self._write_manifest("failed", error=f"{type(e).__name__}: {e}")
            raise
        wall = time.time() - loop_start
        h = hashlib.sha256(rollout_path.read_bytes()).hexdigest()
        rollout_path.chmod(0o444)
        timing = {
            "train_wall_seconds": round(wall, 2),
            "median_step_seconds": round(sorted(step_times)[len(step_times) // 2], 3)
            if step_times else None,
            "rollout_seconds": round(totals["rollout_seconds"], 2),
            "update_seconds": round(totals["update_seconds"], 2),
            "peak_memory_gib": peak_memory_gib(self.device),
        }  # fmt: skip
        if self.guard is not None:
            timing["budget_ledger"] = self.guard.write(
                timing["median_step_seconds"], cfg.steps - self.step, stop_reason
            )
        self._write_manifest(stop_reason or "completed", final_checkpoint=str(last),
                             timing=timing, completions=totals["completions"],
                             rollouts_sha256=h,
                             estimated_cost_usd=round(self.guard.accrued_usd(), 4)
                             if self.guard else 0.0)  # fmt: skip
        return {"status": stop_reason or "completed", "final_checkpoint": str(last),
                "steps": self.step, "updates": self.updates, **timing}  # fmt: skip
