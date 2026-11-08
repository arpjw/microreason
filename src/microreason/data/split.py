"""Semantic fingerprints, prompt normalization and corpus contamination scanning (spec §4)."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable
from fractions import Fraction

from microreason.verify.expr import Bin, Node, Num, fmt_value, linear_coeffs

# ----------------------------------------------------------------------------- fingerprints


def _canon(node: Node) -> str:
    if isinstance(node, Num):
        return fmt_value(node.value)
    if not isinstance(node, Bin):
        raise ValueError("expression fingerprints only cover numeric trees")
    a, b = _canon(node.left), _canon(node.right)
    if node.op in "+*":  # commutative: order-insensitive
        a, b = sorted((a, b))
    return f"({node.op} {a} {b})"


def expression_fingerprint(tree: Node) -> str:
    return "sha256:" + hashlib.sha256(("expr:" + _canon(tree)).encode()).hexdigest()


def equation_fingerprint(lhs: Node, rhs: Node) -> str:
    """Invariant to term order within a side and to swapping the two sides.

    Scaling is deliberately NOT normalized: 5x = 35 and 7x = 49 share a solution but are
    different exercises, and collapsing them would exhaust whole templates.
    """
    sides = sorted(
        (fmt_value(a), fmt_value(b)) for a, b in (linear_coeffs(lhs), linear_coeffs(rhs))
    )
    key = "lineq:" + repr(sides)
    return "sha256:" + hashlib.sha256(key.encode()).hexdigest()


def normalize_prompt(prompt: str) -> str:
    return " ".join(prompt.lower().split())


def answer_type(v: Fraction) -> str:
    return "integer" if v.denominator == 1 else "rational"


# ----------------------------------------------------------------------------- contamination

_UNICODE_OPS = str.maketrans({"−": "-", "–": "-", "×": "*", "÷": "/",
                              "⋅": "*", "·": "*"})  # fmt: skip
_AROUND_OPS = re.compile(r"\s*([+\-*/=()^])\s*")
_SPACES = re.compile(r"\s+")
# A match only counts if it is not glued to surrounding numbers, words or operators.
_BAD_NEIGHBORS = frozenset("abcdefghijklmnopqrstuvwxyz0123456789.+-*/^")


def normalize_math(text: str) -> str:
    t = text.lower().translate(_UNICODE_OPS)
    t = _AROUND_OPS.sub(r"\1", t)
    return _SPACES.sub(" ", t)


class ContaminationScanner:
    """Find which normalized math strings occur verbatim in a text corpus (Aho-Corasick)."""

    def __init__(self, keys: Iterable[str]):
        import ahocorasick

        self.automaton = ahocorasick.Automaton()
        self.n_keys = 0
        for k in set(keys):
            self.automaton.add_word(k, len(k))
            self.n_keys += 1
        if self.n_keys:
            self.automaton.make_automaton()
        self.found: set[str] = set()
        self.docs_scanned = 0
        self.chars_scanned = 0

    def scan(self, text: str) -> None:
        self.docs_scanned += 1
        if not self.n_keys:
            return
        t = normalize_math(text)
        self.chars_scanned += len(t)
        n = len(t)
        for end, length in self.automaton.iter(t):
            start = end - length + 1
            if start > 0 and t[start - 1] in _BAD_NEIGHBORS:
                continue
            if end + 1 < n and t[end + 1] in _BAD_NEIGHBORS:
                continue
            self.found.add(t[start : end + 1])


def iter_corpus_texts(sources_config: str, raw_dir: str) -> Iterable[tuple[str, str]]:
    """Yield (file sha256, text) for every document in the pinned, downloaded source files."""
    from pathlib import Path

    import pyarrow.parquet as pq

    from microreason.data.fetch import local_path, sha256_file
    from microreason.data.sources import load_sources

    for source in load_sources(sources_config).sources:
        for f in source.files:
            path = local_path(Path(raw_dir), source, f)
            if not path.exists():
                raise FileNotFoundError(f"{path} missing; run `microreason data fetch-sources`")
            if sha256_file(path) != f.sha256:
                raise ValueError(f"{path} does not match its pinned sha256")
            pf = pq.ParquetFile(path)
            for batch in pf.iter_batches(columns=[source.text_field], batch_size=4096):
                for text in batch.column(0).to_pylist():
                    if text:
                        yield f.sha256, text
