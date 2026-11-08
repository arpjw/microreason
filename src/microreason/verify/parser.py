"""Parse an untrusted model completion into its single [answer] block (spec §5).

The completion is the assistant text after the [assistant] marker. Text outside the answer
block is ignored for scoring; whether an optional [steps] block is well formed is reported
separately. Anything ambiguous fails closed with a reason code.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

MAX_COMPLETION_CHARS = 4096
MAX_ANSWER_CHARS = 64

ANSWER_OPEN, ANSWER_CLOSE = "[answer]", "[/answer]"
STEPS_OPEN, STEPS_CLOSE = "[steps]", "[/steps]"
EOS = "[eos]"
FORBIDDEN_MARKERS = ("[bos]", "[pad]", "[user]", "[assistant]")

StepsStatus = Literal["absent", "well_formed", "malformed"]


@dataclass(frozen=True)
class ParsedCompletion:
    ok: bool
    reason: str
    answer_text: str | None = None
    steps_status: StepsStatus = "absent"


def _steps_status(text: str, answer_start: int) -> StepsStatus:
    n_open, n_close = text.count(STEPS_OPEN), text.count(STEPS_CLOSE)
    if n_open == 0 and n_close == 0:
        return "absent"
    if n_open != 1 or n_close != 1:
        return "malformed"
    s, e = text.index(STEPS_OPEN), text.index(STEPS_CLOSE)
    if not (s < e < answer_start):
        return "malformed"
    return "well_formed"


def parse_completion(text: object) -> ParsedCompletion:
    if not isinstance(text, str):
        return ParsedCompletion(False, "not_text")
    if len(text) > MAX_COMPLETION_CHARS:
        return ParsedCompletion(False, "completion_too_long")

    body = text.rstrip()
    if body.endswith(EOS):
        body = body[: -len(EOS)]
    if EOS in body or any(m in body for m in FORBIDDEN_MARKERS):
        return ParsedCompletion(False, "forbidden_marker")

    n_open, n_close = body.count(ANSWER_OPEN), body.count(ANSWER_CLOSE)
    if n_open == 0 and n_close == 0:
        return ParsedCompletion(False, "missing_answer")
    if n_open > 1 or n_close > 1:
        return ParsedCompletion(False, "multiple_answer_blocks")
    if n_open != n_close:
        return ParsedCompletion(False, "unbalanced_answer_block")

    start = body.index(ANSWER_OPEN)
    end = body.index(ANSWER_CLOSE)
    if end < start:
        return ParsedCompletion(False, "unbalanced_answer_block")

    steps = _steps_status(body, start)
    if STEPS_OPEN in body and STEPS_CLOSE in body:
        s, e = body.index(STEPS_OPEN), body.index(STEPS_CLOSE)
        if s < start < e:
            return ParsedCompletion(False, "answer_inside_steps", steps_status="malformed")

    inner = body[start + len(ANSWER_OPEN) : end]
    if "[" in inner or "]" in inner:
        return ParsedCompletion(False, "markup_in_answer", steps_status=steps)
    inner = inner.strip()
    if not inner:
        return ParsedCompletion(False, "empty_answer", steps_status=steps)
    if len(inner) > MAX_ANSWER_CHARS:
        return ParsedCompletion(False, "answer_too_long", steps_status=steps)
    return ParsedCompletion(True, "ok", answer_text=inner, steps_status=steps)
