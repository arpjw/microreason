"""Sampling, scoring and response-token log probabilities shared by the gate, GRPO and DPO.

- Completions are sampled with the evaluation generator (`eval.generate`), in the show-work SFT
  format, and scored by the same function the evaluation uses (`eval.score.score_completion`).
  The reward is the primary binary reward (spec §5): 1 only if the response finished with [eos],
  has exactly one valid answer block, and the exact answer is correct; else 0. Verifier errors
  score 0 and are counted separately.
- Log probabilities are computed by teacher forcing over right-padded prompt+completion
  sequences. Only completion tokens (everything generated after [assistant], including the final
  [eos]) count; prompt and padding positions are masked out. Logits are divided by the sampling
  temperature so the probabilities are those of the distribution the samples came from.
"""

from __future__ import annotations

import contextlib
from collections import Counter, defaultdict
from dataclasses import dataclass

import torch

from microreason.eval.generate import eval_prompt_ids, generate
from microreason.eval.score import score_completion
from microreason.tokenizer import MicroTokenizer

ROLLOUT_FORMAT = "sft-show-work-v1"
MALFORMED = ("invalid_format", "truncated")


def autocast(device: torch.device, precision: str):
    if device.type == "cuda" and precision == "bf16":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    if precision not in ("fp32", "bf16"):
        raise ValueError(f"precision {precision} is not supported")
    if device.type == "mps" and precision == "bf16":
        raise ValueError("bf16 autocast on MPS is impractically slow; use precision fp32 locally")
    return contextlib.nullcontext()


@dataclass
class Completion:
    ids: list[int]
    hit_eos: bool
    status: str
    reason: str
    reward: float
    step_credit: float = 0.0

    @property
    def correct(self) -> bool:
        """Exact-answer success. Shaped rewards add at most w < 1, so reward >= 1 iff correct."""
        return self.reward >= 1.0


def passes_filter(group: list[Completion], lo: int, hi: int) -> bool:
    """The GRPO training filter: lo <= correct samples <= hi, and rewards not all equal."""
    n = sum(c.correct for c in group)
    return lo <= n <= hi and len({c.reward for c in group}) > 1


def step_prefix_credit(text: str, gold_steps: list[str]) -> float:
    """Fraction of the gold solution steps the completion reproduces exactly, as a prefix.

    The show-work format numbers steps "1. ...", "2. ..." inside [steps] ... [/steps]
    (`prompts.assistant_ids`). Gold traces are canonical (post-order reductions, fixed equation
    moves), so a step counts only if it is character-identical to the gold step at the same
    position and every earlier step also matched. Extra or repeated steps earn nothing, so the
    credit cannot be raised by padding. Returns a value in [0, 1].
    """
    if not gold_steps or "[steps]" not in text:
        return 0.0
    body = text.split("[steps]", 1)[1].split("[/steps]", 1)[0]
    lines = [ln.strip() for ln in body.strip().split("\n") if ln.strip()]
    n = 0
    for i, (line, gold) in enumerate(zip(lines, gold_steps, strict=False)):
        if line != f"{i + 1}. {gold}":
            break
        n += 1
    return n / len(gold_steps)


def score(record: dict, ids: list[int], hit_eos: bool, tok: MicroTokenizer,
          step_reward_weight: float = 0.0) -> Completion:  # fmt: skip
    text = tok.decode(ids)
    s = score_completion(record, text, hit_eos, ROLLOUT_FORMAT)
    reward = 1.0 if s["status"] == "correct" else 0.0
    credit = 0.0
    if step_reward_weight:
        credit = step_prefix_credit(text, record.get("solution_steps") or [])
        reward += step_reward_weight * credit
    return Completion(ids, hit_eos, s["status"], s["reason"], reward, credit)


@torch.no_grad()
def sample_groups(
    model,
    tok: MicroTokenizer,
    records: list[dict],
    *,
    group_size: int,
    temperature: float,
    max_new_tokens: int,
    generator: torch.Generator,
    batch_size: int,
    precision: str = "fp32",
    use_cache: bool = False,
    step_reward_weight: float = 0.0,
) -> list[list[Completion]]:
    """Sample `group_size` completions per record. Returns one group per record, in order."""
    device = next(model.parameters()).device
    was_training = model.training
    model.eval()
    flat: list[tuple[int, list[int]]] = []
    for i, r in enumerate(records):
        p = eval_prompt_ids(tok, r["prompt"], ROLLOUT_FORMAT)
        flat += [(i, p)] * group_size
    groups: list[list[Completion]] = [[] for _ in records]
    for start in range(0, len(flat), batch_size):
        chunk = flat[start : start + batch_size]
        with autocast(device, precision):
            gens = generate(model, [p for _, p in chunk], max_new_tokens=max_new_tokens,
                            eos_id=tok.eos_id, pad_id=tok.pad_id, temperature=temperature,
                            generator=generator, use_cache=use_cache)  # fmt: skip
        for (i, _), g in zip(chunk, gens, strict=True):
            groups[i].append(score(records[i], g.ids, g.hit_eos, tok, step_reward_weight))
    if was_training:
        model.train()
    return groups


def group_stats(records: list[dict], groups: list[list[Completion]]) -> dict:
    """pass@1, mixed / all-correct / all-wrong group fractions, malformed and verifier-error
    rates, overall and by template."""

    def summarize(pairs):
        n_groups = len(pairs)
        comps = [c for _, g in pairs for c in g]
        if not comps:
            return {}
        succ = [sum(c.correct for c in g) for _, g in pairs]
        size = len(pairs[0][1])
        return {
            "groups": n_groups,
            "completions": len(comps),
            "pass_at_1": sum(c.correct for c in comps) / len(comps),
            "pass_at_k": sum(s > 0 for s in succ) / n_groups,
            "mixed_group_fraction": sum(0 < s < size for s in succ) / n_groups,
            "all_correct_fraction": sum(s == size for s in succ) / n_groups,
            "all_wrong_fraction": sum(s == 0 for s in succ) / n_groups,
            "malformed_fraction": sum(c.status in MALFORMED for c in comps) / len(comps),
            "verifier_error_fraction": sum(c.status == "verifier_error" for c in comps)
            / len(comps),
            "mean_completion_tokens": sum(len(c.ids) for c in comps) / len(comps),
            "status_counts": dict(Counter(c.status for c in comps)),
        }

    pairs = list(zip(records, groups, strict=True))
    by_t: dict[str, list] = defaultdict(list)
    for r, g in pairs:
        by_t[r["template_id"]].append((r, g))
    return {
        "overall": summarize(pairs),
        "by_template": {t: summarize(v) for t, v in sorted(by_t.items())},
    }


def build_sequences(
    prompts: list[list[int]], completions: list[list[int]], pad_id: int
) -> dict[str, torch.Tensor]:
    """Right-padded prompt+completion batch. `response_mask[:, j]` marks targets ids[:, j+1]
    that are completion tokens (so it aligns with next-token log probabilities)."""
    lengths = [len(p) + len(c) for p, c in zip(prompts, completions, strict=True)]
    width = max(lengths)
    b = len(prompts)
    ids = torch.full((b, width), pad_id, dtype=torch.long)
    attn = torch.zeros((b, width), dtype=torch.long)
    resp = torch.zeros((b, width - 1), dtype=torch.bool)
    for i, (p, c) in enumerate(zip(prompts, completions, strict=True)):
        seq = p + c
        ids[i, : len(seq)] = torch.tensor(seq)
        attn[i, : len(seq)] = 1
        # target index j+1 in [len(p), len(seq)) -> logit index j in [len(p)-1, len(seq)-1)
        resp[i, len(p) - 1 : len(seq) - 1] = True
    return {"input_ids": ids, "attention_mask": attn, "response_mask": resp}


def token_logprobs(model, batch: dict[str, torch.Tensor], temperature: float = 1.0) -> torch.Tensor:
    """Per-position log probability of the next token, shape (B, T-1), fp32. Positions outside
    `response_mask` hold 0."""
    ids, attn = batch["input_ids"], batch["attention_mask"]
    logits = model(ids, attention_mask=attn).logits[:, :-1, :].float()
    if temperature != 1.0:
        logits = logits / temperature
    lp = torch.log_softmax(logits, dim=-1).gather(-1, ids[:, 1:, None]).squeeze(-1)
    return lp * batch["response_mask"]
