"""Training-only prompt pool for the RL-stage branches (stream rl1).

Records of the rl1 stream are assigned to disjoint partitions by a hash of their id, so the gate,
GRPO prompts, DPO pair prompts and the SFT control's examples never share a problem:

    gate       first 0.5%   reward-variance gate (spec §6 Stage 5: >= 200 prompts)
    grpo       next 25%     GRPO rollout prompts
    dpo        next 14.5%   DPO pair prompts (pairs come from v2-sft's own samples)
    sft_extra  last 60%     v2-extra-sft supervised examples (gold solutions)

Within a partition records are ordered by sha256 of their id, which is the fixed selection order.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

PARTITIONS: tuple[tuple[str, float], ...] = (
    ("gate", 0.005),
    ("grpo", 0.255),
    ("dpo", 0.40),
    ("sft_extra", 1.0),
)
PARTITION_RULE = (
    "bucket = sha256('rl-partition:' + id)[:8] / 0xFFFFFFFF; gate < 0.005 <= grpo < 0.255 "
    "<= dpo < 0.40 <= sft_extra"
)
ORDER_RULE = "within a partition, sorted by sha256(id)"


def partition_of(record_id: str) -> str:
    bucket = int(hashlib.sha256(("rl-partition:" + record_id).encode()).hexdigest()[:8], 16)
    x = bucket / 0xFFFFFFFF
    for name, upper in PARTITIONS:
        if x < upper:
            return name
    return PARTITIONS[-1][0]


def load_partition(manifest_path: str | Path, partition: str, limit: int | None = None):
    """Return (records, manifest) for one partition in fixed order. Verifies the file hash."""
    from microreason.data.stream import SPLIT, iter_stream_dicts

    if partition not in {p for p, _ in PARTITIONS}:
        raise ValueError(f"unknown partition {partition!r}")
    manifest = json.loads(Path(manifest_path).read_text())
    if manifest.get("split") != SPLIT:
        raise ValueError(f"{manifest_path} is not a training-only stream manifest")
    records = [r for r in iter_stream_dicts(manifest) if partition_of(r["id"]) == partition]
    records.sort(key=lambda r: hashlib.sha256(r["id"].encode()).hexdigest())
    if limit is not None:
        if len(records) < limit:
            raise ValueError(f"partition {partition} has {len(records)} records, need {limit}")
        records = records[:limit]
    return records, manifest
