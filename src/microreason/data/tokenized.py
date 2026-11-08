"""Pre-tokenized document datasets stored as memory-mapped uint16 token streams.

Layout of a dataset directory:
    tokens.bin      all documents' tokens, concatenated (uint16; vocab is 8,192)
    offsets.npy     int64 array of length n_docs + 1; doc i is tokens[offsets[i]:offsets[i+1]]
    manifest.json   source, tokenizer hash, document/token counts and file checksums

Documents are stored without [bos]/[eos]; the packing stream adds separators.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from pathlib import Path

import numpy as np

TOKENS_FILE = "tokens.bin"
OFFSETS_FILE = "offsets.npy"
MANIFEST_FILE = "manifest.json"
DTYPE = np.uint16


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def write_tokenized(
    out_dir: str | Path,
    docs: Iterable[list[int]],
    meta: dict,
) -> dict:
    out = Path(out_dir)
    if (out / MANIFEST_FILE).exists():
        raise FileExistsError(f"{out} already contains a tokenized dataset; refusing to overwrite")
    out.mkdir(parents=True, exist_ok=True)
    tmp_tokens = out / (TOKENS_FILE + ".partial")
    offsets = [0]
    with tmp_tokens.open("wb") as f:
        for ids in docs:
            arr = np.asarray(ids, dtype=np.int64)
            if arr.size and (arr.min() < 0 or arr.max() >= np.iinfo(DTYPE).max):
                raise ValueError("token id out of uint16 range")
            f.write(arr.astype(DTYPE).tobytes())
            offsets.append(offsets[-1] + arr.size)
    np.save(out / (OFFSETS_FILE + ".partial.npy"), np.asarray(offsets, dtype=np.int64))
    (out / (OFFSETS_FILE + ".partial.npy")).replace(out / OFFSETS_FILE)
    tmp_tokens.replace(out / TOKENS_FILE)
    manifest = {
        "schema_version": 1,
        **meta,
        "documents": len(offsets) - 1,
        "tokens": offsets[-1],
        "dtype": "uint16",
        "files": {
            TOKENS_FILE: _sha256(out / TOKENS_FILE),
            OFFSETS_FILE: _sha256(out / OFFSETS_FILE),
        },
    }
    (out / MANIFEST_FILE).write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


class TokenizedDataset:
    def __init__(self, directory: str | Path, verify: bool = False):
        self.dir = Path(directory)
        self.manifest = json.loads((self.dir / MANIFEST_FILE).read_text())
        if verify:
            for name, digest in self.manifest["files"].items():
                if _sha256(self.dir / name) != digest:
                    raise ValueError(f"{self.dir / name} does not match its manifest checksum")
        self.offsets = np.load(self.dir / OFFSETS_FILE)
        n_tokens = int(self.offsets[-1])
        self.tokens = (
            np.memmap(self.dir / TOKENS_FILE, dtype=DTYPE, mode="r", shape=(n_tokens,))
            if n_tokens
            else np.zeros(0, dtype=DTYPE)
        )
        self.name = self.manifest.get("name", self.dir.name)

    def __len__(self) -> int:
        return len(self.offsets) - 1

    @property
    def n_tokens(self) -> int:
        return int(self.offsets[-1])

    def doc(self, i: int) -> np.ndarray:
        return self.tokens[self.offsets[i] : self.offsets[i + 1]]

    @property
    def fingerprint(self) -> str:
        return self.manifest["files"][TOKENS_FILE]


def tokenize_general_text(tok, split: str, out_dir: str | Path, batch: int = 2048) -> dict:
    from microreason.data.text_split import iter_split_texts

    def gen():
        buf: list[str] = []
        for text in iter_split_texts("configs/data_sources.yaml", "data/raw", split):
            buf.append(text)
            if len(buf) == batch:
                yield from tok.encode_batch(buf)
                buf = []
        if buf:
            yield from tok.encode_batch(buf)

    split_manifest = json.loads(Path("data/manifests/general_text_split_v1.json").read_text())
    return write_tokenized(
        out_dir,
        gen(),
        {
            "name": f"general_text_{split}",
            "source": "general_text",
            "split": split,
            "split_manifest_val_digest": split_manifest["val_digest_set_sha256"],
            "tokenizer_sha256": tok.sha256,
            "format": "raw document text",
        },
    )


def tokenize_math(tok, split: str, out_dir: str | Path) -> dict:
    from microreason.data.build import load_split
    from microreason.tokenizer import math_plain_text

    manifest = json.loads(Path(f"data/manifests/math_{split}_v1.json").read_text())
    if split != "train" and split != "val":
        raise ValueError("only train and val math splits may be tokenized for training/eval loss")
    recs = [r.model_dump() for r in load_split(manifest)]
    ids = tok.encode_batch([math_plain_text(r) for r in recs])
    return write_tokenized(
        out_dir,
        iter(ids),
        {
            "name": f"math_{split}",
            "source": "math",
            "split": split,
            "math_manifest_sha256": manifest["sha256"],
            "tokenizer_sha256": tok.sha256,
            "format": "math_plain_text (prompt, numbered steps, 'Answer: x')",
        },
    )
