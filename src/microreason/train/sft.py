"""Supervised fine-tuning stage (spec §6, Stage 3).

Examples come from the verified v1 math train split, so every target answer and step trace has
passed the independent checks. Selection and format assignment are fixed before training:

- Records are ordered by sha256 of their id; the first `num_examples` are used.
- Each example is show-work or direct by a separate hash of its id, at `show_work_fraction`.
- Prompts use the central `sft-v1` format; loss covers assistant tokens only (prompts and
  padding are -100), and an example that does not fit `sequence_length` is an error, never
  truncated.

Batches are right-padded per micro-batch to the longest example in it. The stream visits a
seeded permutation each epoch and saves its position for exact resume.
"""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

import torch

from microreason.config import ResolvedConfig
from microreason.prompts import IGNORE_INDEX, PROMPT_FORMAT_VERSION, Example, build_record_example
from microreason.tokenizer import MicroTokenizer

SELECTION_RULE = "records sorted by sha256(id), first num_examples"
MODE_RULE = "show_work if sha256('mode:' + id) bucket < show_work_fraction else direct"


def _bucket(text: str) -> float:
    return int(hashlib.sha256(text.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF


def mode_for(record_id: str, show_work_fraction: float) -> str:
    return "show_work" if _bucket("mode:" + record_id) < show_work_fraction else "direct"


def select_records(manifest_path: str, split: str, n: int) -> tuple[list[dict], dict]:
    from microreason.data.build import load_split

    manifest = json.loads(Path(manifest_path).read_text())
    if manifest["split"] != split:
        raise ValueError(f"{manifest_path} is split '{manifest['split']}', need '{split}'")
    records = [r.model_dump(mode="json") for r in load_split(manifest)]
    records.sort(key=lambda r: hashlib.sha256(r["id"].encode()).hexdigest())
    if len(records) < n:
        raise ValueError(f"{manifest_path} has {len(records)} records, need {n}")
    return records[:n], manifest


def pad_examples(examples: list[Example], pad_id: int) -> dict[str, torch.Tensor]:
    width = max(len(e.input_ids) for e in examples)
    ids = torch.full((len(examples), width), pad_id, dtype=torch.long)
    labels = torch.full((len(examples), width), IGNORE_INDEX, dtype=torch.long)
    mask = torch.zeros((len(examples), width), dtype=torch.long)
    for i, e in enumerate(examples):
        n = len(e.input_ids)
        ids[i, :n] = torch.tensor(e.input_ids)
        labels[i, :n] = torch.tensor(e.labels)
        mask[i, :n] = 1
    return {"input_ids": ids, "labels": labels, "attention_mask": mask}


class SFTStream:
    def __init__(self, items: list[tuple[str, Example]], seed: int, pad_id: int, name: str):
        if not items:
            raise ValueError("SFT stream needs examples")
        self.items = items
        self.seed = seed
        self.pad_id = pad_id
        self.name = name
        self.fingerprint = hashlib.sha256(
            "\n".join(f"{rid}:{len(e.input_ids)}" for rid, e in items).encode()
        ).hexdigest()
        self.epoch = 0
        self.pos = 0
        self.digest = hashlib.sha256(b"sft-stream-v1").hexdigest()
        self._order_cache: tuple[int, list[int]] | None = None

    def _order(self) -> list[int]:
        if self._order_cache is None or self._order_cache[0] != self.epoch:
            order = list(range(len(self.items)))
            random.Random(f"{self.seed}|{self.name}|{self.epoch}").shuffle(order)
            self._order_cache = (self.epoch, order)
        return self._order_cache[1]

    def next_batch(self, n: int) -> dict[str, torch.Tensor]:
        chosen: list[Example] = []
        while len(chosen) < n:
            if self.pos == len(self.items):
                self.epoch += 1
                self.pos = 0
            rid, ex = self.items[self._order()[self.pos]]
            self.pos += 1
            chosen.append(ex)
            self.digest = hashlib.sha256(f"{self.digest}|{rid}".encode()).hexdigest()
        return pad_examples(chosen, self.pad_id)

    def mixture_report(self) -> dict:
        return {self.name: {"examples": len(self.items), "epoch": self.epoch, "position": self.pos}}

    def state_dict(self) -> dict:
        return {"version": 1, "kind": "sft", "name": self.name, "fingerprint": self.fingerprint,
                "seed": self.seed, "epoch": self.epoch, "pos": self.pos,
                "digest": self.digest}  # fmt: skip

    def load_state_dict(self, state: dict) -> None:
        mine = (self.name, self.fingerprint, self.seed)
        if (state.get("name"), state.get("fingerprint"), state.get("seed")) != mine:
            raise ValueError("SFT stream state does not match these examples or settings")
        self.epoch, self.pos, self.digest = state["epoch"], state["pos"], state["digest"]
        self._order_cache = None


def build_sft_trainer(
    resolved: ResolvedConfig, run_dir: str | Path, device: str, budget_guard=None
):
    from microreason.checkpoint import load_checkpoint
    from microreason.train.common import Trainer

    stage = resolved.stage
    cfg = stage.sft
    if stage.stage != "sft" or cfg is None:
        raise NotImplementedError(f"stage '{stage.stage}' is not built by the SFT builder")
    if stage.parent is None:
        raise ValueError("SFT needs a parent checkpoint")
    tok = MicroTokenizer.load(stage.tokenizer)
    if tok.vocab_size != resolved.model.vocab_size:
        raise ValueError("tokenizer vocab size does not match the model config")

    records, train_manifest = select_records(cfg.train_manifest, "train", cfg.num_examples)
    items: list[tuple[str, Example]] = []
    counts = {"show_work": 0, "direct": 0}
    for r in records:
        mode = mode_for(r["id"], cfg.show_work_fraction)
        counts[mode] += 1
        items.append((r["id"], build_record_example(tok, r, mode, max_len=stage.sequence_length)))
    stream = SFTStream(items, stage.seed, tok.pad_id, "sft_math_train_v1")

    micro = stage.micro_batch_size or stage.sequences_per_step
    val_records, val_manifest = select_records(cfg.val_manifest, "val", cfg.val_examples_per_format)
    eval_sets = {}
    for mode in ("show_work", "direct"):
        exs = [
            build_record_example(tok, r, mode, max_len=stage.sequence_length) for r in val_records
        ]
        eval_sets[f"sft_val_{mode}"] = [pad_examples(exs[i : i + micro], tok.pad_id)
                                        for i in range(0, len(exs), micro)]  # fmt: skip

    model, _, parent_manifest = load_checkpoint(stage.parent, "cpu")
    lengths = [len(e.input_ids) for _, e in items]
    provenance = {
        "tokenizer_sha256": tok.sha256,
        "train_datasets": {stream.name: stream.fingerprint},
        "sft_data": {
            "train_manifest_sha256": train_manifest["sha256"],
            "val_manifest_sha256": val_manifest["sha256"],
            "selection_rule": SELECTION_RULE,
            "mode_rule": MODE_RULE,
            "prompt_format": PROMPT_FORMAT_VERSION,
            "examples": len(items),
            "by_mode": counts,
            "tokens_per_example": {
                "mean": round(sum(lengths) / len(lengths), 1),
                "max": max(lengths),
            },  # fmt: skip
        },
        "validation_datasets": {k: f"{len(val_records)} val records" for k in eval_sets},
        "parent_config_hash": parent_manifest["config_hash"],
        "parent_step": parent_manifest["step"],
    }
    return Trainer(resolved, run_dir, stream, eval_sets, device, provenance, model, budget_guard)
