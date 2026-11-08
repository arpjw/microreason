"""Batched generation for evaluation (spec §7).

Prompts are left-padded into one batch; the model derives positions from the attention mask, so
a padded prompt sees exactly the same positions as it would alone. Greedy decoding is the
primary protocol; sampling uses a seeded generator so sampled pass@k is reproducible.

With `use_cache=True` the prompt is run once and each new token reuses the stored keys and
values (`model.KVCache`). Positions and attention masks are the same as the uncached path's, so
the two give the same tokens up to floating-point rounding. Evaluation keeps the uncached path
(protocol eval-v1); RL rollouts may use the cache because their loss recomputes log
probabilities by teacher forcing.

Prompt formats are versioned:
  plain-v1          [bos] "{prompt}\\n"  -> the model continues "1. ...\\nAnswer: x[eos]"
                    (the format base and mid-trained checkpoints saw in pretraining)
  sft-show-work-v1  [bos][user]{prompt} Show your work.[eos][assistant]
  sft-direct-v1     [bos][user]{prompt} Give only the answer.[eos][assistant]
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch

from microreason.prompts import prompt_ids
from microreason.tokenizer import MicroTokenizer

PromptFormat = Literal["plain-v1", "sft-show-work-v1", "sft-direct-v1"]
FORMATS: tuple[str, ...] = ("plain-v1", "sft-show-work-v1", "sft-direct-v1")


def eval_prompt_ids(tok: MicroTokenizer, prompt: str, fmt: str) -> list[int]:
    if fmt == "plain-v1":
        return [tok.bos_id, *tok.encode(prompt + "\n")]
    if fmt == "sft-show-work-v1":
        return prompt_ids(tok, prompt, "show_work")
    if fmt == "sft-direct-v1":
        return prompt_ids(tok, prompt, "direct")
    raise ValueError(f"unknown prompt format {fmt!r}")


@dataclass(frozen=True)
class Generation:
    ids: list[int]  # generated ids, including the final [eos] when one was produced
    hit_eos: bool


@torch.no_grad()
def generate(
    model,
    prompts: list[list[int]],
    *,
    max_new_tokens: int,
    eos_id: int,
    pad_id: int,
    temperature: float = 0.0,
    generator: torch.Generator | None = None,
    use_cache: bool = False,
) -> list[Generation]:
    """Generate for one batch of prompts. temperature 0 means greedy."""
    from microreason.model import KVCache

    if not prompts:
        return []
    device = next(model.parameters()).device
    longest = max(map(len, prompts))
    if longest + max_new_tokens > model.config.max_seq_len:
        raise ValueError("prompt plus max_new_tokens exceeds the model context")
    b = len(prompts)
    ids = torch.full((b, longest), pad_id, dtype=torch.long, device=device)
    mask = torch.zeros((b, longest), dtype=torch.long, device=device)
    for i, p in enumerate(prompts):
        ids[i, longest - len(p) :] = torch.tensor(p, device=device)
        mask[i, longest - len(p) :] = 1

    finished = torch.zeros(b, dtype=torch.bool, device=device)
    steps: list[torch.Tensor] = []
    cache = KVCache(model.config.n_layers, longest + max_new_tokens) if use_cache else None
    for step in range(max_new_tokens):
        if cache is None:
            logits = model(ids, attention_mask=mask).logits[:, -1, :].float()
        else:
            new = ids if step == 0 else ids[:, -1:]
            logits = model(new, attention_mask=mask, cache=cache).logits[:, -1, :].float()
        if temperature == 0.0:
            nxt = logits.argmax(dim=-1)
        else:
            probs = torch.softmax(logits / temperature, dim=-1)
            nxt = torch.multinomial(probs, 1, generator=generator).squeeze(-1)
        nxt = torch.where(finished, torch.full_like(nxt, pad_id), nxt)
        steps.append(nxt)
        finished |= nxt == eos_id
        ids = torch.cat([ids, nxt[:, None]], dim=1)
        mask = torch.cat([mask, (~finished | (nxt == eos_id)).long()[:, None]], dim=1)
        if bool(finished.all()):
            break
    # Each row keeps its tokens up to and including its first [eos] (all of them if none):
    # exactly the tokens generated while the row was unfinished.
    out: list[list[int]] = []
    for row in torch.stack(steps, dim=1).tolist():
        cut = row.index(eos_id) + 1 if eos_id in row else len(row)
        out.append(row[:cut])
    return [Generation(o, bool(o) and o[-1] == eos_id) for o in out]


def generate_all(
    model,
    prompts: list[list[int]],
    *,
    batch_size: int,
    **kwargs,
) -> list[Generation]:
    """Generate in length-sorted batches (less padding), returning results in input order."""
    order = sorted(range(len(prompts)), key=lambda i: (len(prompts[i]), i))
    results: list[Generation | None] = [None] * len(prompts)
    for start in range(0, len(order), batch_size):
        idx = order[start : start + batch_size]
        for i, g in zip(idx, generate(model, [prompts[i] for i in idx], **kwargs), strict=True):
            results[i] = g
    return results  # type: ignore[return-value]
