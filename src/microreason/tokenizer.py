"""Byte-level BPE tokenizer shared by every stage (spec §3).

Special markers have reserved single-token IDs and are only ever inserted by prompt assembly as
IDs. Plain text that happens to contain "[answer]" is encoded as ordinary bytes, so user or
model text cannot forge a marker.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, PositiveInt
from tokenizers import Tokenizer, decoders, models, normalizers, pre_tokenizers, trainers

SPECIAL_TOKENS = (
    "[pad]",
    "[bos]",
    "[eos]",
    "[user]",
    "[assistant]",
    "[steps]",
    "[/steps]",
    "[answer]",
    "[/answer]",
)
NORMALIZATION = "replace CRLF and CR with LF, then Unicode NFC"
TOKENIZER_FILE = "tokenizer.json"
MANIFEST_FILE = "tokenizer_manifest.json"


class TokenizerConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    version: str = "v1"
    vocab_size: PositiveInt = 8192
    min_frequency: PositiveInt = 2
    sources_config: str = "configs/data_sources.yaml"
    raw_dir: str = "data/raw"
    text_select_per_100k: int = Field(default=1300, gt=0, le=100_000)
    math_manifest: str = "data/manifests/math_train_v1.json"
    math_format: Literal["plain"] = "plain"
    # "merge" (v1): numbers are merged by BPE like any text. "individual" (v2, protocol v3):
    # every digit is split off before byte-level pre-tokenization, so each digit is exactly one
    # token in every position and numbers of any length share the same digit tokens.
    digits: Literal["merge", "individual"] = "merge"
    out_dir: str = "artifacts/tokenizer/v1"


def load_tokenizer_config(path: str | Path) -> TokenizerConfig:
    with Path(path).open("r", encoding="utf-8") as f:
        return TokenizerConfig.model_validate(yaml.safe_load(f))


def _normalizer() -> normalizers.Normalizer:
    return normalizers.Sequence(
        [normalizers.Replace("\r\n", "\n"), normalizers.Replace("\r", "\n"), normalizers.NFC()]
    )


def normalize_text(text: str) -> str:
    return _normalizer().normalize_str(text)


def math_plain_text(record: dict) -> str:
    """Math record as plain text (no markers) for tokenizer training."""
    steps = "\n".join(f"{i}. {s}" for i, s in enumerate(record["solution_steps"], 1))
    return f"{record['prompt']}\n{steps}\nAnswer: {record['final_answer']}"


def collect_training_texts(cfg: TokenizerConfig) -> tuple[list[str], dict]:
    """Training-only text: a hash-selected sample of general-text train docs, then math train."""
    from microreason.data.build import load_split
    from microreason.data.text_split import iter_split_texts, text_digest

    general = [
        t
        for t in iter_split_texts(cfg.sources_config, cfg.raw_dir, "train")
        if int(text_digest(t)[8:16], 16) % 100_000 < cfg.text_select_per_100k
    ]
    manifest = json.loads(Path(cfg.math_manifest).read_text())
    if manifest["split"] != "train":
        raise ValueError("tokenizer math source must be the train split")
    math = [math_plain_text(r.model_dump()) for r in load_split(manifest)]
    g_chars, m_chars = sum(map(len, general)), sum(map(len, math))
    stats = {
        "general_text_docs": len(general),
        "general_text_chars": g_chars,
        "general_text_split": "train",
        "math_records": len(math),
        "math_chars": m_chars,
        "math_manifest_sha256": manifest["sha256"],
        "math_char_fraction": round(m_chars / (g_chars + m_chars), 4),
    }
    return general + math, stats


def iter_training_texts(cfg: TokenizerConfig) -> Iterator[str]:
    yield from collect_training_texts(cfg)[0]


def pre_tokenizer(digits: str = "merge") -> pre_tokenizers.PreTokenizer:
    byte_level = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True)
    if digits == "merge":
        return byte_level
    return pre_tokenizers.Sequence([pre_tokenizers.Digits(individual_digits=True), byte_level])


def pre_tokenization_description(digits: str = "merge") -> str:
    base = "ByteLevel(add_prefix_space=False, use_regex=True)"
    return base if digits == "merge" else f"Digits(individual_digits=True) then {base}"


def train_tokenizer(cfg: TokenizerConfig, texts: Iterable[str]) -> Tokenizer:
    tok = Tokenizer(models.BPE(unk_token=None, byte_fallback=False))
    tok.normalizer = _normalizer()
    tok.pre_tokenizer = pre_tokenizer(cfg.digits)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=cfg.vocab_size,
        min_frequency=cfg.min_frequency,
        special_tokens=list(SPECIAL_TOKENS),
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=False,
    )
    tok.train_from_iterator(texts, trainer=trainer)
    tok.encode_special_tokens = True
    return tok


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class MicroTokenizer:
    """Thin wrapper that fixes encode/decode behavior for the whole project."""

    def __init__(self, tok: Tokenizer, path: Path | None = None):
        tok.encode_special_tokens = True  # markers in text are bytes, never special IDs
        self.tok = tok
        self.path = path
        self.special = {t: tok.token_to_id(t) for t in SPECIAL_TOKENS}
        if any(v is None for v in self.special.values()):
            raise ValueError("tokenizer is missing a reserved special token")
        self.pad_id = self.special["[pad]"]
        self.bos_id = self.special["[bos]"]
        self.eos_id = self.special["[eos]"]

    @classmethod
    def load(cls, directory: str | Path) -> MicroTokenizer:
        directory = Path(directory)
        return cls(Tokenizer.from_file(str(directory / TOKENIZER_FILE)), directory)

    @property
    def vocab_size(self) -> int:
        return self.tok.get_vocab_size()

    @property
    def sha256(self) -> str | None:
        return file_sha256(self.path / TOKENIZER_FILE) if self.path else None

    def encode(self, text: str) -> list[int]:
        return self.tok.encode(text, add_special_tokens=False).ids

    def encode_batch(self, texts: list[str]) -> list[list[int]]:
        return [e.ids for e in self.tok.encode_batch(texts, add_special_tokens=False)]

    def decode(self, ids: list[int]) -> str:
        """Decode, rendering special IDs as their marker text (what the verifier parses)."""
        out: list[str] = []
        run: list[int] = []
        inverse = {v: k for k, v in self.special.items()}
        for i in ids:
            if i in inverse:
                if run:
                    out.append(self.tok.decode(run, skip_special_tokens=False))
                    run = []
                out.append(inverse[i])
            else:
                run.append(i)
        if run:
            out.append(self.tok.decode(run, skip_special_tokens=False))
        return "".join(out)


def save_tokenizer(tok: Tokenizer, cfg: TokenizerConfig, stats: dict) -> Path:
    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / TOKENIZER_FILE
    tok.save(str(path))
    wrapped = MicroTokenizer(tok, out)
    manifest = {
        "schema_version": 1,
        "version": cfg.version,
        "tokenizer_sha256": file_sha256(path),
        "vocab_size": tok.get_vocab_size(),
        "model": "byte-level BPE",
        "normalization": NORMALIZATION,
        "pre_tokenization": pre_tokenization_description(cfg.digits),
        "special_tokens": wrapped.special,
        "special_tokens_matched_in_text": False,
        "config": cfg.model_dump(mode="json"),
        "training_data": stats,
    }
    (out / MANIFEST_FILE).write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return path
