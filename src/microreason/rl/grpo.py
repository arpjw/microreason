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

**Dynamic sampling (protocol rl-v2, `dynamic_sampling: true`).** Instead of a fixed
`prompts_per_step` prompts, the step draws chunks of `sample_chunk_prompts` new prompts, samples
each chunk from the current policy, and keeps only mixed groups (some success, some failure)
until `prompts_per_step` are collected or `max_sample_rounds` chunks have been drawn; surplus
mixed groups beyond the target are dropped. Every sampled completion is still recorded, with
`used_in_loss` false for dropped ones. Dropped groups would have zero advantage, so the only term
they lose is their KL contribution. Prompts are consumed in order and never repeat.

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
    passes_filter,
    sample_groups,
    token_logprobs,
)

log = get_logger("microreason.rl.grpo")


def group_advantages(rewards: torch.Tensor, eps: float = 1e-6, mode: str = "group_std",
                     batch_rows: torch.Tensor | None = None) -> torch.Tensor:  # fmt: skip
    """rewards: (groups, group_size) -> advantages; zero for zero-variance groups.

    group_std (v1-v3): (r - group mean) / group std. group_mean (Dr. GRPO): r - group mean.
    batch_std: (r - group mean) / std of the rewards in `batch_rows` (the trained groups;
    default all). reward (RFT): A = r, no baseline.
    """
    if mode == "reward":
        return rewards.clone()
    mean = rewards.mean(dim=1, keepdim=True)
    std = rewards.std(dim=1, unbiased=False, keepdim=True)
    centered = rewards - mean
    if mode == "group_std":
        adv = centered / (std + eps)
    elif mode == "group_mean":
        adv = centered
    elif mode == "batch_std":
        sel = rewards if batch_rows is None else rewards[batch_rows]
        bstd = sel.std(unbiased=False) if sel.numel() > 1 else torch.tensor(0.0)
        adv = centered / (bstd + eps)
    else:
        raise ValueError(f"unknown advantage mode {mode!r}")
    return torch.where(std > 0, adv, torch.zeros_like(adv))


def grpo_loss(
    new_lp: torch.Tensor,
    old_lp: torch.Tensor,
    ref_lp: torch.Tensor,
    mask: torch.Tensor,
    adv: torch.Tensor,
    clip_eps: float,
    kl_beta: float,
    *,
    clip_eps_high: float | None = None,
    aggregation: str = "sequence_mean",
    max_len: int | None = None,
    denom: float | None = None,
) -> tuple[torch.Tensor, dict]:
    """All log-prob tensors (B, T) fp32, mask (B, T) bool, adv (B,). Returns (loss, stats).

    sequence_mean: sum over completions of each completion's token mean, / denom (default B).
    token_constant: sum over every completion token, / (denom x max_len) (Dr. GRPO), with
    denom defaulting to B. Passing the whole update's completion count as `denom` lets
    micro-batches accumulate into exactly the full-batch loss. The clip range is
    [1 - clip_eps, 1 + clip_eps_high] (DAPO clip-higher when clip_eps_high > clip_eps).
    """
    hi = clip_eps if clip_eps_high is None else clip_eps_high
    m = mask.float()
    n_tok = m.sum(dim=1).clamp(min=1.0)
    log_ratio = (new_lp - old_lp) * m
    ratio = torch.exp(log_ratio)
    a = adv[:, None]
    surr = torch.minimum(ratio * a, torch.clamp(ratio, 1 - clip_eps, 1 + hi) * a)
    d = (ref_lp - new_lp) * m
    kl = torch.exp(d) - d - 1
    per_tok = -(surr - kl_beta * kl) * m
    n = float(new_lp.shape[0]) if denom is None else float(denom)
    if aggregation == "sequence_mean":
        loss = (per_tok.sum(dim=1) / n_tok).sum() / n
    elif aggregation == "token_constant":
        if not max_len:
            raise ValueError("token_constant aggregation needs max_len")
        loss = per_tok.sum() / (n * max_len)
    else:
        raise ValueError(f"unknown loss aggregation {aggregation!r}")
    with torch.no_grad():
        clipped = (((ratio - 1) > hi) | ((1 - ratio) > clip_eps)) & mask
        stats = {
            "kl": float(((kl * m).sum(dim=1) / n_tok).mean()),
            "clip_fraction": float(clipped.sum() / m.sum().clamp(min=1.0)),
            "ratio_mean": float((ratio * m).sum() / m.sum().clamp(min=1.0)),
        }
    return loss, stats


def _objective_description(cfg: GRPOConfig) -> str:
    if cfg.objective == "grpo" and (
        cfg.advantage, cfg.loss_aggregation, cfg.clip_eps_high, cfg.step_reward_weight,
        cfg.mask_truncated) == ("group_std", "sequence_mean", None, 0.0, False):  # fmt: skip
        return "GRPO, clipped ratio, k3 KL to frozen parent, sequence-mean tokens"
    lo, hi = cfg.filter_bounds()
    head = ("RFT (on-policy rejection-sampling fine-tuning: A = reward, no baseline)"
            if cfg.objective == "rft" else f"GRPO, advantage {cfg.advantage}")  # fmt: skip
    clip = f"clip [1-{cfg.clip_eps}, 1+{cfg.clip_eps_high or cfg.clip_eps}]"
    reward = ("binary exact-answer reward" if not cfg.step_reward_weight else
              f"exact answer + {cfg.step_reward_weight} x gold step-prefix fraction")  # fmt: skip
    return (
        f"{head}; {clip}; k3 KL beta {cfg.kl_beta} to frozen parent; {cfg.loss_aggregation} "
        f"aggregation; {reward}; filter {lo}..{hi} correct of {cfg.sampling.group_size}"
        + ("; truncated masked" if cfg.mask_truncated else "")
    )


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
        # dynamic sampling consumes a variable number of prompts, so it keeps the whole order
        self.prompts = [pool[i] for i in (order if cfg.dynamic_sampling else order[:need])]
        self.cursor = 0
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
            "objective": _objective_description(cfg),
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

    def _sample(self, recs: list[dict]):
        s = self.cfg.sampling
        return sample_groups(self.policy, self.tok, recs, group_size=s.group_size,
                             temperature=s.temperature, max_new_tokens=s.max_new_tokens,
                             generator=self.gen, batch_size=s.generation_batch_size,
                             precision=self.cfg.precision, use_cache=s.kv_cache,
                             step_reward_weight=self.cfg.step_reward_weight)  # fmt: skip

    def _keep_group(self, g: list) -> bool:
        """The training filter: lo <= correct samples <= hi and rewards not all equal."""
        return passes_filter(g, *self.cfg.filter_bounds())

    def _collect(self) -> tuple[list[dict], list[list], list[bool], int]:
        """Sampled records, their groups, which groups train, and the number of rounds."""
        cfg = self.cfg
        if not cfg.dynamic_sampling:
            recs = self.prompts[self.cursor : self.cursor + cfg.prompts_per_step]
            self.cursor += len(recs)
            groups = self._sample(recs)
            return recs, groups, [True] * len(recs), 1
        recs, groups, use = [], [], []
        kept, rounds = 0, 0
        while kept < cfg.prompts_per_step and rounds < cfg.max_sample_rounds:
            chunk = self.prompts[self.cursor : self.cursor + cfg.sample_chunk_prompts]
            if not chunk:
                raise RuntimeError("grpo partition exhausted during dynamic sampling")
            self.cursor += len(chunk)
            rounds += 1
            for r, g in zip(chunk, self._sample(chunk), strict=True):
                take = self._keep_group(g) and kept < cfg.prompts_per_step
                kept += take
                recs.append(r)
                groups.append(g)
                use.append(take)
        return recs, groups, use, rounds

    def train_step(self, rollout_file) -> dict:
        cfg, s = self.cfg, self.cfg.sampling
        lr = _lr_at(self.step, cfg)
        for g in self.optimizer.param_groups:
            g["lr"] = lr
        t0 = time.time()
        recs, groups, use, rounds = self._collect()
        t_roll = time.time() - t0
        rewards_all = torch.tensor([[c.reward for c in g] for g in groups])
        mode = "reward" if cfg.objective == "rft" else cfg.advantage
        adv_all = group_advantages(rewards_all, cfg.adv_eps, mode,
                                   batch_rows=torch.tensor(use, dtype=torch.bool))  # fmt: skip
        from microreason.eval.generate import eval_prompt_ids
        from microreason.rl.rollout import ROLLOUT_FORMAT

        gen_settings = {"temperature": s.temperature, "max_new_tokens": s.max_new_tokens,
                        "group_size": s.group_size, "format": ROLLOUT_FORMAT,
                        "kv_cache": s.kv_cache}  # fmt: skip
        train_idx = [gi for gi, u in enumerate(use) if u]
        flat = [(gi, c) for gi in train_idx for c in groups[gi]]
        adv = torch.tensor([float(adv_all[gi, j]) for gi in train_idx
                            for j in range(len(groups[gi]))])  # fmt: skip
        t1 = time.time()
        old_cpu = mask_cpu = None
        if flat:
            prompt_cache = {gi: eval_prompt_ids(self.tok, recs[gi]["prompt"], ROLLOUT_FORMAT)
                            for gi in train_idx}  # fmt: skip
            # an empty completion still needs one target position; generate() emits >= 1 id
            batch = build_sequences([prompt_cache[gi] for gi, _ in flat],
                                    [c.ids for _, c in flat], self.tok.pad_id)  # fmt: skip
            batch = {k: v.to(self.device) for k, v in batch.items()}
            old_lp = self._chunked_logprobs(self.policy, batch)
            ref_lp = self._chunked_logprobs(self.ref, batch)
            old_cpu, mask_cpu = old_lp.cpu(), batch["response_mask"].cpu()
            if cfg.mask_truncated:
                keep = torch.tensor([c.status != "truncated" for _, c in flat],
                                    device=self.device)  # fmt: skip
                batch["response_mask"] = batch["response_mask"] & keep[:, None]
        # immutable rollout record: every sampled completion, trained on or not
        row_of = {(gi, j): n for n, (gi, j) in enumerate(
            (gi, j) for gi in train_idx for j in range(len(groups[gi])))}  # fmt: skip
        for gi, (r, g) in enumerate(zip(recs, groups, strict=True)):
            for j, c in enumerate(g):
                row = row_of.get((gi, j))
                lp, a = None, 0.0
                if row is not None:
                    lp = [round(x, 5) for x in old_cpu[row][mask_cpu[row]].tolist()]
                    a = round(float(adv_all[gi, j]), 6)
                rollout_file.write(json.dumps({
                    "prompt_id": r["id"], "template_id": r["template_id"],
                    "step": self.step, "policy_updates_before": self.updates,
                    "parent_checkpoint_id": self.parent_id, "token_ids": c.ids,
                    "old_logprobs": lp, "n_tokens": len(c.ids), "used_in_loss": row is not None,
                    "reward": c.reward, "status": c.status, "reason": c.reason,
                    "hit_eos": c.hit_eos, "advantage": a,
                    "generation": gen_settings,
                }, sort_keys=True) + "\n")  # fmt: skip
        stats = defaultdict(float)
        grad_norms = []
        if flat:
            m = batch["response_mask"]
            stats["sampled_token_nll"] = float(-(old_lp * m).sum() / m.sum().clamp(min=1))
        if not flat:
            log.warning("step %d: no mixed groups after %d rounds; no update", self.step, rounds)
        else:
            adv = adv.to(self.device)
            perm = torch.randperm(len(flat), generator=self.shuffle_rng)
            self._update(batch, old_lp, ref_lp, adv, perm, stats, grad_norms)
        t_upd = time.time() - t1
        self.step += 1
        return self._step_metrics(lr, recs, groups, use, rounds, rewards_all, stats, grad_norms,
                                  t_roll, t_upd)  # fmt: skip

    @torch.no_grad()
    def _chunked_logprobs(self, model, batch: dict) -> torch.Tensor:
        """Teacher-forced completion log probabilities in micro-batches (memory-bounded)."""
        cfg = self.cfg
        n = batch["input_ids"].shape[0]
        out = []
        for i in range(0, n, cfg.update_micro_batch):
            sub = {k: v[i : i + cfg.update_micro_batch] for k, v in batch.items()}
            with autocast(self.device, cfg.precision):
                out.append(token_logprobs(model, sub, cfg.sampling.temperature))
        return torch.cat(out)

    def _update(self, batch, old_lp, ref_lp, adv, perm, stats, grad_norms) -> None:
        """One optimizer update per minibatch; each minibatch accumulates gradients over
        micro-batches of `update_micro_batch` completions, normalized by the minibatch size, so
        the update equals a single full-minibatch step."""
        cfg, s = self.cfg, self.cfg.sampling
        for chunk in perm.chunk(cfg.minibatches_per_step):
            idx_all = chunk.to(self.device)
            self.optimizer.zero_grad(set_to_none=True)
            loss_total = 0.0
            st_acc: dict[str, float] = defaultdict(float)
            for idx in idx_all.split(cfg.update_micro_batch):
                sub = {k: v[idx] for k, v in batch.items()}
                with autocast(self.device, cfg.precision):
                    new_lp = token_logprobs(self.policy, sub, s.temperature)
                loss, st = grpo_loss(new_lp, old_lp[idx], ref_lp[idx], sub["response_mask"],
                                     adv[idx], cfg.clip_eps, cfg.kl_beta,
                                     clip_eps_high=cfg.clip_eps_high,
                                     aggregation=cfg.loss_aggregation,
                                     max_len=s.max_new_tokens, denom=len(idx_all))  # fmt: skip
                loss.backward()
                loss_total += float(loss.detach())
                w = len(idx) / len(idx_all)
                for k, v in st.items():
                    st_acc[k] += v * w
            gn = torch.nn.utils.clip_grad_norm_(self.policy.parameters(), cfg.grad_clip_norm)
            gn = float(gn)
            if not (math.isfinite(loss_total) and math.isfinite(gn)):
                raise RuntimeError(f"non-finite GRPO loss/grad at step {self.step}")
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)
            self.updates += 1
            grad_norms.append(gn)
            stats["loss"] += loss_total / cfg.minibatches_per_step
            for k, v in st_acc.items():
                stats[k] += v / cfg.minibatches_per_step

    def _step_metrics(self, lr, recs, groups, use, rounds, rewards, stats, grad_norms, t_roll,
                      t_upd) -> dict:  # fmt: skip
        """Reward and group statistics are over every sampled group (the policy's behaviour);
        loss statistics are over the groups trained on."""
        s = self.cfg.sampling
        succ = torch.tensor([sum(c.correct for c in g) for g in groups])
        k = s.group_size
        kept = [self._keep_group(g) for g in groups]
        comps = [c for g in groups for c in g]
        by_family: dict[str, list[float]] = defaultdict(list)
        for r, g in zip(recs, groups, strict=True):
            by_family[r["family"]] += [float(c.correct) for c in g]
        return {
            "lr": lr,
            "sampled_prompts": len(recs),
            "trained_groups": sum(use),
            "sample_rounds": rounds,
            "mean_reward": float(rewards.mean()),
            "pass_rate": float(succ.float().mean() / k),
            "filter_yield": sum(kept) / len(kept),
            "mean_step_credit": sum(c.step_credit for c in comps) / len(comps),
            "mixed_group_fraction": float(((succ > 0) & (succ < k)).float().mean()),
            "all_correct_fraction": float((succ == k).float().mean()),
            "all_wrong_fraction": float((succ == 0).float().mean()),
            "malformed_fraction": sum(c.status in MALFORMED for c in comps) / len(comps),
            "truncated_fraction": sum(c.status == "truncated" for c in comps) / len(comps),
            "verifier_errors": sum(c.status == "verifier_error" for c in comps),
            "mean_response_tokens": sum(len(c.ids) for c in comps) / len(comps),
            "grad_norm": max(grad_norms) if grad_norms else 0.0,
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
                    totals["completions"] += m["sampled_prompts"] * cfg.sampling.group_size
                    totals["trained_completions"] += m["trained_groups"] * cfg.sampling.group_size
                    totals["rollout_seconds"] += m["rollout_seconds"]
                    totals["update_seconds"] += m["update_seconds"]
                    self._event("train", step_seconds=round(step_times[-1], 3),
                                peak_memory_gib=peak_memory_gib(self.device), **m)  # fmt: skip
                    if self.step % cfg.log_every_steps == 0 or self.step == cfg.steps:
                        log.info("step %d/%d reward %.3f mixed %.2f kl %.4f clip %.3f len %.1f "
                                 "%.1fs", self.step, cfg.steps, m["mean_reward"],
                                 m["mixed_group_fraction"], m.get("kl", 0.0),
                                 m.get("clip_fraction", 0.0),
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
        self.provenance["pool"]["prompts_used"] = self.cursor
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
                             trained_completions=totals["trained_completions"],
                             rollouts_sha256=h,
                             estimated_cost_usd=round(self.guard.accrued_usd(), 4)
                             if self.guard else 0.0)  # fmt: skip
        return {"status": stop_reason or "completed", "final_checkpoint": str(last),
                "steps": self.step, "updates": self.updates, **timing}  # fmt: skip
