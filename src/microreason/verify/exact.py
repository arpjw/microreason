"""Exact answer verification and the primary GRPO reward (spec §5).

The verifier is a pure function of (expected answer, verifier_spec, completion). It never looks
at logits, train-set membership or a judge model. Verifier errors fail closed and are reported
separately from wrong answers.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from fractions import Fraction
from typing import Literal

from microreason.verify.expr import (
    MAX_DIGITS,
    evaluate,
    fmt_value,
    parse_equation,
    parse_expression,
)
from microreason.verify.parser import StepsStatus, parse_completion

# v1 answer grammar: integer or p/q, optional leading minus on either part, ASCII digits only.
_ANSWER_RE = re.compile(r"(-?)([0-9]+)(?:/(-?)([0-9]+))?")

Status = Literal["correct", "wrong", "invalid_format", "verifier_error"]


class AnswerFormatError(ValueError):
    def __init__(self, code: str):
        super().__init__(code)
        self.code = code


def parse_answer_value(text: str) -> Fraction:
    """Parse an answer string under the v1 grammar into an exact rational."""
    m = _ANSWER_RE.fullmatch(text)
    if m is None:
        if re.fullmatch(r"-?[0-9]*\.[0-9]+", text):
            raise AnswerFormatError("decimal_not_allowed")
        raise AnswerFormatError("invalid_number")
    num_sign, num, den_sign, den = m.groups()
    if len(num) > MAX_DIGITS or (den is not None and len(den) > MAX_DIGITS):
        raise AnswerFormatError("number_too_long")
    p = int(num) * (-1 if num_sign else 1)
    if den is None:
        return Fraction(p)
    q = int(den) * (-1 if den_sign else 1)
    if q == 0:
        raise AnswerFormatError("zero_denominator")
    return Fraction(p, q)


@dataclass(frozen=True)
class VerifyResult:
    status: Status
    reason: str
    parsed_answer: str | None = None
    canonical_answer: str | None = None
    steps_status: StepsStatus = "absent"

    @property
    def correct(self) -> bool:
        return self.status == "correct"


def _check_spec(spec: dict, expected: Fraction, value: Fraction, is_equal: bool) -> str | None:
    """Cross-check against the problem itself. Returns a reason on internal disagreement."""
    kind = spec.get("kind")
    text = spec.get("text")
    if kind == "expression":
        if evaluate(parse_expression(text)) != expected:
            return "expected_disagrees_with_expression"
        return None
    if kind == "linear_equation":
        lhs, rhs = parse_equation(text)
        if evaluate(lhs, expected) != evaluate(rhs, expected):
            return "expected_fails_substitution"
        if (evaluate(lhs, value) == evaluate(rhs, value)) != is_equal:
            return "substitution_disagrees"
        return None
    return "unknown_kind"


def verify_answer_text(
    final_answer: str,
    verifier_spec: dict,
    answer_text: str,
    steps_status: StepsStatus = "absent",
) -> VerifyResult:
    """Score an answer string already extracted from a completion."""
    try:
        expected = parse_answer_value(final_answer)
    except Exception:
        return VerifyResult("verifier_error", "bad_expected_answer")
    try:
        try:
            value = parse_answer_value(answer_text)
        except AnswerFormatError as e:
            return VerifyResult("invalid_format", e.code, answer_text, steps_status=steps_status)
        is_equal = value == expected
        problem = _check_spec(verifier_spec, expected, value, is_equal)
        if problem:
            return VerifyResult("verifier_error", problem, answer_text)
        return VerifyResult(
            "correct" if is_equal else "wrong",
            "match" if is_equal else "mismatch",
            answer_text,
            fmt_value(value),
            steps_status,
        )
    except Exception as e:  # fail closed
        return VerifyResult("verifier_error", f"exception:{type(e).__name__}")


def verify_completion(final_answer: str, verifier_spec: dict, completion: object) -> VerifyResult:
    try:
        parse_answer_value(final_answer)
    except Exception:
        return VerifyResult("verifier_error", "bad_expected_answer")
    try:
        parsed = parse_completion(completion)
    except Exception as e:  # fail closed
        return VerifyResult("verifier_error", f"exception:{type(e).__name__}")
    if not parsed.ok:
        return VerifyResult("invalid_format", parsed.reason, steps_status=parsed.steps_status)
    return verify_answer_text(final_answer, verifier_spec, parsed.answer_text, parsed.steps_status)


def primary_reward(final_answer: str, verifier_spec: dict, completion: object) -> float:
    """R_primary = 1 if the response parses and its exact answer is correct, else 0."""
    return 1.0 if verify_completion(final_answer, verifier_spec, completion).correct else 0.0
