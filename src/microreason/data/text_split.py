"""Deterministic train/validation split of the general-text corpus.

A document's split depends only on the sha256 of its own text, so the assignment is stable under
reordering and identical documents always land in the same split. Validation documents are used
to measure general-text loss (forgetting) and never for tokenizer or model training.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

from microreason.data.split import iter_corpus_texts

SPLIT_VERSION = "v1"
VAL_PER_MILLE = 20  # 2% of documents


def text_digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def text_split(text: str, digest: str | None = None) -> str:
    d = digest or text_digest(text)
    return "val" if int(d[:8], 16) % 1000 < VAL_PER_MILLE else "train"


def iter_split_texts(sources_config: str, raw_dir: str, split: str):
    for _, text in iter_corpus_texts(sources_config, raw_dir):
        if text_split(text) == split:
            yield text


def build_text_split_manifest(sources_config: str, raw_dir: str, out: str | Path) -> dict:
    counts: Counter = Counter()
    chars: Counter = Counter()
    val_digests: list[str] = []
    files: set[str] = set()
    for file_sha, text in iter_corpus_texts(sources_config, raw_dir):
        files.add(file_sha)
        d = text_digest(text)
        s = text_split(text, d)
        counts[s] += 1
        chars[s] += len(text)
        if s == "val":
            val_digests.append(d)
    manifest = {
        "schema_version": 1,
        "dataset": "general_text",
        "version": SPLIT_VERSION,
        "rule": f"val if int(sha256(text)[:8], 16) % 1000 < {VAL_PER_MILLE} else train",
        "source_files": sorted(files),
        "documents": dict(counts),
        "chars": dict(chars),
        "val_digest_set_sha256": hashlib.sha256(
            "\n".join(sorted(val_digests)).encode()
        ).hexdigest(),
    }
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest
