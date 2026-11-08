"""DPO preference branch from v2-sft (spec §6 Stage 4A).

**Pairs.** For prompts of the `dpo` partition (training-only, in fixed order), v2-sft samples a
group of completions (show-work format, temperature 0.8). A group with at least one correct and
one incorrect completion yields one pair: chosen = its first correct completion, rejected = its
first incorrect one. Correctness is the primary binary reward, so no human or learned labels
are used. Prompts are processed until `target_pairs` pairs exist or `max_pair_prompts` prompts
have been sampled. A pair is held out for discrimination checks when
sha256('dpo-heldout:' + prompt_id) falls below `heldout_fraction`; both responses of a prompt
always share a split.

**Loss.** Policy initialized from v2-sft, with v2-sft frozen as the reference.
loss = -log sigmoid(beta * ((pi_c - ref_c) - (pi_r - ref_r))), where each term is the sum of
response-token log probabilities (temperature 1; prompt and padding masked). AdamW, linear
warmup then constant LR, `epochs` passes in a seeded order.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import random
import time
from datetime import UTC, datetime
from pathlib import Path

import torch
import torch.nn.functional as F

from microreason.logging_utils import get_logger
from microreason.rl.config import DPOConfig, ResolvedRL
from microreason.rl.rollout import autocast, build_sequences, sample_groups, token_logprobs

log = get_logger("microreason.rl.dpo")


def is_heldout(prompt_id: str, fraction: float) -> bool:
    b = int(hashlib.sha256(("dpo-heldout:" + prompt_id).encode()).hexdigest()[:8], 16)
    return b / 0xFFFFFFFF < fraction


def dpo_loss(pc, pr, rc, rr, beta: float) -> tuple[torch.Tensor, dict]:
    """Sequence log probs (B,) for policy chosen/rejected and reference chosen/rejected."""
    margin = (pc - rc) - (pr - rr)
    loss = -F.logsigmoid(beta * margin).mean()
    with torch.no_grad():
        stats = {"reward_accuracy": float((margin > 0).float().mean()),
                 "margin": float(margin.mean() * beta)}  # fmt: skip
    return loss, stats


def make_pairs(model, tok, records, cfg: DPOConfig, device) -> tuple[list[dict], dict]:
    s = cfg.sampling
    gen = torch.Generator(device=device).manual_seed(cfg.seed)
    pairs: list[dict] = []
    used = 0
    chunk = max(1, s.generation_batch_size // s.group_size) * 4
    mixed = 0
    for start in range(0, min(len(records), cfg.max_pair_prompts), chunk):
        batch = records[start : min(start + chunk, cfg.max_pair_prompts)]
        groups = sample_groups(model, tok, batch, group_size=s.group_size,
                               temperature=s.temperature, max_new_tokens=s.max_new_tokens,
                               generator=gen, batch_size=s.generation_batch_size,
                               precision=cfg.precision)  # fmt: skip
        for r, g in zip(batch, groups, strict=True):
            used += 1
            good = [c for c in g if c.reward == 1.0]
            bad = [c for c in g if c.reward == 0.0]
            if good and bad:
                mixed += 1
                pairs.append({"prompt_id": r["id"], "prompt": r["prompt"],
                              "template_id": r["template_id"], "family": r["family"],
                              "chosen_ids": good[0].ids, "rejected_ids": bad[0].ids,
                              "rejected_status": bad[0].status,
                              "heldout": is_heldout(r["id"], cfg.heldout_fraction)})  # fmt: skip
            if len(pairs) >= cfg.target_pairs:
                break
        if len(pairs) >= cfg.target_pairs:
            break
        log.info("dpo pairs: %d from %d prompts", len(pairs), used)
    return pairs, {"prompts_sampled": used, "pairs": len(pairs),
                   "mixed_group_fraction": mixed / max(1, used)}  # fmt: skip


def _seq_logps(model, tok, pairs, which: str, device, precision) -> torch.Tensor:
    from microreason.eval.generate import eval_prompt_ids
    from microreason.rl.rollout import ROLLOUT_FORMAT

    prompts = [eval_prompt_ids(tok, p["prompt"], ROLLOUT_FORMAT) for p in pairs]
    batch = build_sequences(prompts, [p[which] for p in pairs], tok.pad_id)
    batch = {k: v.to(device) for k, v in batch.items()}
    with autocast(device, precision):
        return token_logprobs(model, batch).sum(dim=1)


class DPOTrainer:
    def __init__(self, resolved: ResolvedRL, run_dir: str | Path, device: str, guard=None,
                 pool: list[dict] | None = None):  # fmt: skip
        from microreason.checkpoint import MANIFEST, load_checkpoint
        from microreason.tokenizer import MicroTokenizer
        from microreason.train.common import _code_revision, param_groups

        cfg = resolved.stage
        if not isinstance(cfg, DPOConfig):
            raise ValueError("DPOTrainer needs a dpo config")
        if cfg.parent is None:
            raise ValueError("DPO needs a parent checkpoint (v2-sft)")
        if cfg.budget.paid_compute_allowed and guard is None:
            raise ValueError("paid compute runs need a budget guard")
        self.cfg, self.resolved, self.guard = cfg, resolved, guard
        self.run_dir = Path(run_dir)
        if (self.run_dir / "pairs.jsonl").exists():
            raise FileExistsError(f"{self.run_dir} already holds a DPO run")
        self.device = torch.device(device)
        if self.device.type == "cuda":
            torch.set_float32_matmul_precision("high")
        torch.manual_seed(cfg.seed)
        self.tok = MicroTokenizer.load(cfg.tokenizer)
        self.policy, _, parent_manifest = load_checkpoint(cfg.parent, self.device)
        self.ref = copy.deepcopy(self.policy).eval()
        for p in self.ref.parameters():
            p.requires_grad_(False)
        self.optimizer = torch.optim.AdamW(param_groups(self.policy, cfg.weight_decay),
                                           lr=cfg.lr, betas=cfg.betas, foreach=False)  # fmt: skip
        if pool is None:
            from microreason.rl.pool import load_partition

            pool, m = load_partition(cfg.pool_manifest, "dpo")
            pool_sha = m["sha256"]
        else:
            pool_sha = "in-memory (tests)"
        self.pool = pool
        self.provenance = {
            "stage": "dpo", "config_hash": resolved.config_hash(), "parent": cfg.parent,
            "parent_checkpoint_id": "sha256:" + hashlib.sha256(
                (Path(cfg.parent) / MANIFEST).read_bytes()).hexdigest(),
            "parent_config_hash": parent_manifest.get("config_hash"),
            "tokenizer_sha256": self.tok.sha256, "code_revision": _code_revision(),
            "pool": {"manifest": cfg.pool_manifest, "sha256": pool_sha, "partition": "dpo"},
        }  # fmt: skip
        self.step = 0

    def _event(self, kind: str, **data) -> None:
        with (self.run_dir / "metrics.jsonl").open("a") as f:
            f.write(json.dumps({"event": kind, "step": self.step, **data}, sort_keys=True) + "\n")

    def _write_manifest(self, status: str, **extra) -> None:
        path = self.run_dir / "run_manifest.json"
        prev = json.loads(path.read_text()) if path.exists() else {}
        m = {**prev, "status": status, "resolved_config": self.resolved.model_dump(mode="json"),
             "provenance": self.provenance, "device": str(self.device),
             "steps_completed": self.step, "updated_at": datetime.now(UTC).isoformat(),
             **extra}  # fmt: skip
        m.setdefault("started_at", m["updated_at"])
        path.write_text(json.dumps(m, indent=2, sort_keys=True) + "\n")

    @torch.no_grad()
    def discrimination(self, pairs: list[dict], bs: int = 64) -> dict:
        """Held-out pair accuracy: does the policy rank chosen above rejected (vs reference)?"""
        if not pairs:
            return {"pairs": 0}
        self.policy.eval()
        acc, margin, raw = 0, 0.0, 0
        for i in range(0, len(pairs), bs):
            ch = pairs[i : i + bs]
            pc = _seq_logps(
                self.policy, self.tok, ch, "chosen_ids", self.device, self.cfg.precision
            )
            pr = _seq_logps(self.policy, self.tok, ch, "rejected_ids", self.device,
                            self.cfg.precision)  # fmt: skip
            rc = _seq_logps(self.ref, self.tok, ch, "chosen_ids", self.device, self.cfg.precision)
            rr = _seq_logps(self.ref, self.tok, ch, "rejected_ids", self.device, self.cfg.precision)
            m = (pc - rc) - (pr - rr)
            acc += int((m > 0).sum())
            margin += float(m.sum())
            raw += int((pc > pr).sum())
        self.policy.train()
        n = len(pairs)
        return {"pairs": n, "implicit_reward_accuracy": acc / n,
                "mean_beta_margin": self.cfg.beta * margin / n,
                "policy_prefers_chosen": raw / n}  # fmt: skip

    def train(self) -> dict:
        from microreason.checkpoint import save_checkpoint

        cfg = self.cfg
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self._write_manifest("running")
        t0 = time.time()
        if self.guard is not None and self.guard.check(None, 0, 1):
            self._write_manifest("stopped_budget_ceiling")
            return {"status": "stopped_budget_ceiling"}
        pairs, pair_stats = make_pairs(self.policy, self.tok, self.pool, cfg, self.device)
        t_pairs = time.time() - t0
        with (self.run_dir / "pairs.jsonl").open("x") as f:
            for p in pairs:
                f.write(json.dumps(p, sort_keys=True) + "\n")
        pairs_sha = hashlib.sha256((self.run_dir / "pairs.jsonl").read_bytes()).hexdigest()
        train = [p for p in pairs if not p["heldout"]]
        held = [p for p in pairs if p["heldout"]]
        if not train:
            raise RuntimeError("no DPO training pairs: v2-sft produced no mixed groups")
        before = self.discrimination(held)
        self._event("heldout_before", **before)
        steps_per_epoch = math.ceil(len(train) / cfg.batch_pairs)
        total = steps_per_epoch * cfg.epochs
        warm = math.ceil(cfg.warmup_fraction * total)
        self.policy.train()
        step_times: list[float] = []
        stop_reason = None
        t_train = time.time()
        for epoch in range(cfg.epochs):
            order = list(range(len(train)))
            random.Random(f"{cfg.seed}|dpo|{epoch}").shuffle(order)
            for i in range(0, len(order), cfg.batch_pairs):
                if self.guard is not None:
                    recent = sorted(step_times[-5:])
                    est = recent[len(recent) // 2] if recent else None
                    stop_reason = self.guard.check(est, len(step_times), total - self.step)
                    if stop_reason:
                        break
                s0 = time.time()
                lr = cfg.lr * (self.step + 1) / warm if self.step < warm else cfg.lr
                for g in self.optimizer.param_groups:
                    g["lr"] = lr
                ch = [train[j] for j in order[i : i + cfg.batch_pairs]]
                pc = _seq_logps(self.policy, self.tok, ch, "chosen_ids", self.device,
                                cfg.precision)  # fmt: skip
                pr = _seq_logps(self.policy, self.tok, ch, "rejected_ids", self.device,
                                cfg.precision)  # fmt: skip
                with torch.no_grad():
                    rc = _seq_logps(self.ref, self.tok, ch, "chosen_ids", self.device,
                                    cfg.precision)  # fmt: skip
                    rr = _seq_logps(self.ref, self.tok, ch, "rejected_ids", self.device,
                                    cfg.precision)  # fmt: skip
                loss, st = dpo_loss(pc, pr, rc, rr, cfg.beta)
                loss.backward()
                gn = float(torch.nn.utils.clip_grad_norm_(self.policy.parameters(),
                                                          cfg.grad_clip_norm))  # fmt: skip
                if not (math.isfinite(float(loss.detach())) and math.isfinite(gn)):
                    raise RuntimeError(f"non-finite DPO loss/grad at step {self.step}")
                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)
                self.step += 1
                step_times.append(time.time() - s0)
                if self.step % cfg.log_every_steps == 0 or self.step == total:
                    self._event("train", loss=float(loss.detach()), grad_norm=gn, lr=lr, **st)
                    log.info("dpo step %d/%d loss %.4f acc %.2f", self.step, total,
                             float(loss.detach()), st["reward_accuracy"])  # fmt: skip
            if stop_reason:
                break
        train_s = time.time() - t_train
        after = self.discrimination(held)
        self._event("heldout_after", **after)
        state = {"step": self.step, "torch_rng": torch.get_rng_state()}
        ckpt = save_checkpoint(self.run_dir, self.step, self.policy, self.optimizer, state,
                               {**self.provenance, "tokens": 0})  # fmt: skip
        from collections import Counter

        timing = {"pair_generation_seconds": round(t_pairs, 2),
                  "train_seconds": round(train_s, 2),
                  "total_wall_seconds": round(time.time() - t0, 2)}  # fmt: skip
        if self.guard is not None:
            timing["budget_ledger"] = self.guard.write(None, total - self.step, stop_reason)
        result = {
            "status": stop_reason or "completed", "final_checkpoint": str(ckpt),
            "pairs": {**pair_stats, "train": len(train), "heldout": len(held),
                      "sha256": pairs_sha,
                      "by_template": dict(Counter(p["template_id"] for p in pairs)),
                      "rejected_status": dict(Counter(p["rejected_status"] for p in pairs))},
            "heldout_before": before, "heldout_after": after, "steps": self.step,
            "timing": timing,
        }  # fmt: skip
        self._write_manifest(result["status"], **{k: v for k, v in result.items()
                                                  if k != "status"})  # fmt: skip
        return result
