"""Deterministic, resumable packed token stream over a weighted mixture of datasets.

Packing policy (declared for spec §6): each document is wrapped as [bos] doc [eos] and
documents are concatenated into fixed-length sequences. Attention may cross document
boundaries; there is no padding, so every position except the first of each sequence is a
prediction target.

Mixture policy: the next document comes from the source whose processed-token share is furthest
below its target weight (ties go to the earlier source). This is deterministic, needs no random
draws and tracks the target ratio closely by processed tokens. Within a source, documents are
visited in a seeded permutation per epoch, so no document is repeated or skipped within an
epoch and a resume continues exactly where the stream stopped.
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass

import torch

from microreason.data.tokenized import TokenizedDataset


@dataclass
class Source:
    name: str
    dataset: TokenizedDataset
    weight: float


class PackedStream:
    """Packed mixture stream; optionally follows a curriculum.

    A curriculum is a list of phases (until_tokens, weights), one weight per source in source
    order. Within a phase the deficit rule tracks that phase's weights using only tokens drawn
    during the phase. Phase changes are recorded in `phase_log` (and in saved state).
    """

    def __init__(
        self,
        sources: list[Source],
        seq_len: int,
        seed: int,
        bos_id: int,
        eos_id: int,
        schedule: list[tuple[int, list[float]]] | None = None,
    ):
        if not sources:
            raise ValueError("need at least one source")
        if any(s.weight <= 0 for s in sources) or any(len(s.dataset) == 0 for s in sources):
            raise ValueError("every source needs a positive weight and at least one document")
        total = sum(s.weight for s in sources)
        self.sources = sources
        self.targets = [s.weight / total for s in sources]
        self.schedule = None
        if schedule:
            bounds = [u for u, _ in schedule]
            if bounds != sorted(bounds) or len(set(bounds)) != len(bounds):
                raise ValueError("curriculum phases must have increasing token bounds")
            for _, w in schedule:
                if len(w) != len(sources) or any(x < 0 for x in w) or sum(w) <= 0:
                    raise ValueError("each phase needs one non-negative weight per source")
            self.schedule = [(u, [x / sum(w) for x in w]) for u, w in schedule]
        self.phase = 0
        self.phase_tokens = [0] * len(sources)
        self.phase_log: list[dict] = []
        self.seq_len = seq_len
        self.seed = seed
        self.bos, self.eos = bos_id, eos_id
        self.epoch = [0] * len(sources)
        self.pos = [0] * len(sources)
        self.tokens = [0] * len(sources)
        self.docs = [0] * len(sources)
        self.buffer: list[int] = []
        self.digest = hashlib.sha256(b"packed-stream-v1").hexdigest()
        self._perm_cache: dict[tuple[int, int], list[int]] = {}

    # ------------------------------------------------------------------ ordering

    def _perm(self, k: int) -> list[int]:
        key = (k, self.epoch[k])
        if key not in self._perm_cache:
            self._perm_cache = {kk: v for kk, v in self._perm_cache.items() if kk[0] != k}
            order = list(range(len(self.sources[k].dataset)))
            random.Random(f"{self.seed}|{self.sources[k].name}|{self.epoch[k]}").shuffle(order)
            self._perm_cache[key] = order
        return self._perm_cache[key]

    def _current_targets(self) -> tuple[list[float], list[int]]:
        if self.schedule is None:
            return self.targets, self.tokens
        processed = sum(self.tokens)
        while self.phase < len(self.schedule) - 1 and processed >= self.schedule[self.phase][0]:
            self.phase += 1
            self.phase_tokens = [0] * len(self.sources)
            self.phase_log.append({"phase": self.phase, "at_tokens": processed})
        return self.schedule[self.phase][1], self.phase_tokens

    def _pick_source(self) -> int:
        targets, counts = self._current_targets()
        total = sum(counts)
        if total == 0:
            return max(range(len(self.sources)), key=lambda k: (targets[k], -k))
        deficits = [targets[k] - counts[k] / total for k in range(len(self.sources))]
        return max(range(len(self.sources)), key=lambda k: (deficits[k], -k))

    def _next_document(self) -> list[int]:
        k = self._pick_source()
        src = self.sources[k]
        idx = self._perm(k)[self.pos[k]]
        self.pos[k] += 1
        if self.pos[k] == len(src.dataset):
            self.epoch[k] += 1
            self.pos[k] = 0
        toks = [self.bos, *src.dataset.doc(idx).tolist(), self.eos]
        self.tokens[k] += len(toks)
        self.phase_tokens[k] += len(toks)
        self.docs[k] += 1
        self.digest = hashlib.sha256(f"{self.digest}|{src.name}:{idx}".encode()).hexdigest()
        return toks

    # ------------------------------------------------------------------ batches

    def next_sequence(self) -> list[int]:
        while len(self.buffer) < self.seq_len:
            self.buffer.extend(self._next_document())
        seq, self.buffer = self.buffer[: self.seq_len], self.buffer[self.seq_len :]
        return seq

    def next_batch(self, batch_size: int) -> torch.Tensor:
        return torch.tensor([self.next_sequence() for _ in range(batch_size)], dtype=torch.long)

    def mixture_report(self) -> dict:
        total = sum(self.tokens) or 1
        report: dict = {
            s.name: {
                "target": round(self.targets[k], 4),
                "actual": round(self.tokens[k] / total, 4),
                "documents": self.docs[k],
                "epoch": self.epoch[k],
            }
            for k, s in enumerate(self.sources)
        }
        if self.schedule is not None:
            report["curriculum"] = {
                "phases": [
                    {"until_tokens": u, "weights": {s.name: round(w[k], 4)
                                                    for k, s in enumerate(self.sources)}}
                    for u, w in self.schedule
                ],
                "current_phase": self.phase,
                "phase_log": self.phase_log,
            }  # fmt: skip
        return report

    # ------------------------------------------------------------------ state

    def state_dict(self) -> dict:
        return {
            "version": 2,
            "schedule": [[u, list(w)] for u, w in self.schedule] if self.schedule else None,
            "phase": self.phase,
            "phase_tokens": list(self.phase_tokens),
            "phase_log": list(self.phase_log),
            "sources": [
                {"name": s.name, "fingerprint": s.dataset.fingerprint, "weight": s.weight}
                for s in self.sources
            ],
            "seq_len": self.seq_len,
            "seed": self.seed,
            "epoch": list(self.epoch),
            "pos": list(self.pos),
            "tokens": list(self.tokens),
            "docs": list(self.docs),
            "buffer": list(self.buffer),
            "digest": self.digest,
        }

    def load_state_dict(self, state: dict) -> None:
        mine = [
            {"name": s.name, "fingerprint": s.dataset.fingerprint, "weight": s.weight}
            for s in self.sources
        ]
        if (
            state["sources"] != mine
            or state["seq_len"] != self.seq_len
            or state["seed"] != self.seed
            or state.get("schedule") != self._schedule_state()
        ):
            raise ValueError("stream state does not match this stream's sources or settings")
        self.epoch = list(state["epoch"])
        self.pos = list(state["pos"])
        self.tokens = list(state["tokens"])
        self.docs = list(state["docs"])
        self.buffer = list(state["buffer"])
        self.digest = state["digest"]
        self.phase = state.get("phase", 0)
        self.phase_tokens = list(state.get("phase_tokens", [0] * len(self.sources)))
        self.phase_log = list(state.get("phase_log", []))
        self._perm_cache = {}

    def _schedule_state(self) -> list | None:
        return [[u, list(w)] for u, w in self.schedule] if self.schedule else None


def fixed_eval_batches(
    dataset: TokenizedDataset, n_sequences: int, seq_len: int, batch_size: int, bos: int, eos: int
) -> list[torch.Tensor]:
    """The first n packed sequences of a dataset under a fixed seed: identical across runs."""
    stream = PackedStream([Source(dataset.name, dataset, 1.0)], seq_len, 0, bos, eos)
    seqs = [stream.next_sequence() for _ in range(n_sequences)]
    return [
        torch.tensor(seqs[i : i + batch_size], dtype=torch.long)
        for i in range(0, len(seqs), batch_size)
    ]
