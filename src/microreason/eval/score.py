"""Scoring rows and aggregate statistics (spec §7).

Every prompt stays in the denominator. A completion is one of:
  correct | wrong | invalid_format | truncated | verifier_error
`truncated` means generation hit max_new_tokens without [eos]; it is a failure even if an answer
appeared, because the protocol requires a finished response.
"""

from __future__ import annotations

import math
import random
from collections import Counter, defaultdict
from dataclasses import dataclass

from microreason.verify.exact import VerifyResult, verify_answer_text, verify_completion
from microreason.verify.parser import EOS, FORBIDDEN_MARKERS, MAX_ANSWER_CHARS

PLAIN_ANSWER_PREFIX = "Answer:"


def verify_plain(final_answer: str, verifier_spec: dict, completion: str) -> VerifyResult:
    """plain-v1: exactly one line starting with 'Answer:'; the rest of that line is the answer."""
    body = completion.rstrip()
    if body.endswith(EOS):
        body = body[: -len(EOS)]
    if EOS in body or any(m in body for m in FORBIDDEN_MARKERS) or "[answer]" in body:
        return VerifyResult("invalid_format", "forbidden_marker")
    lines = [ln for ln in body.split("\n") if ln.strip().startswith(PLAIN_ANSWER_PREFIX)]
    if not lines:
        return VerifyResult("invalid_format", "missing_answer")
    if len(lines) > 1:
        return VerifyResult("invalid_format", "multiple_answer_blocks")
    answer = lines[0].strip()[len(PLAIN_ANSWER_PREFIX) :].strip()
    if not answer:
        return VerifyResult("invalid_format", "empty_answer")
    if len(answer) > MAX_ANSWER_CHARS:
        return VerifyResult("invalid_format", "answer_too_long")
    return verify_answer_text(final_answer, verifier_spec, answer)


def score_completion(record: dict, completion: str, hit_eos: bool, fmt: str) -> dict:
    spec = record["verifier_spec"]
    if fmt == "plain-v1":
        r = verify_plain(record["final_answer"], spec, completion)
    else:
        r = verify_completion(record["final_answer"], spec, completion)
    status, reason = r.status, r.reason
    if not hit_eos:
        status, reason = "truncated", f"no_eos:{r.status}"
    return {
        "status": status,
        "reason": reason,
        "parsed_answer": r.parsed_answer,
        "canonical_answer": r.canonical_answer,
        "steps_status": r.steps_status,
    }


# ----------------------------------------------------------------------------- statistics


def wilson_interval(successes: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


@dataclass(frozen=True)
class Rate:
    numerator: int
    denominator: int

    def as_dict(self) -> dict:
        lo, hi = wilson_interval(self.numerator, self.denominator)
        value = self.numerator / self.denominator if self.denominator else None
        return {
            "numerator": self.numerator,
            "denominator": self.denominator,
            "value": value,
            "wilson95": [lo, hi],
        }


def summarize_greedy(rows: list[dict]) -> dict:
    n = len(rows)
    statuses = Counter(r["status"] for r in rows)
    correct = statuses["correct"]
    valid = statuses["correct"] + statuses["wrong"]
    by_family: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    by_level: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for r in rows:
        for table, key in ((by_family, r["family"]), (by_level, str(r["difficulty"]))):
            table[key][0] += r["status"] == "correct"
            table[key][1] += 1
    toks = sorted(r["generated_tokens"] for r in rows) or [0]
    return {
        "prompts": n,
        "exact_accuracy": Rate(correct, n).as_dict(),
        "valid_format_rate": Rate(valid, n).as_dict(),
        "status_counts": dict(sorted(statuses.items())),
        "accuracy_by_family": {k: Rate(*v).as_dict() for k, v in sorted(by_family.items())},
        "accuracy_by_difficulty": {k: Rate(*v).as_dict() for k, v in sorted(by_level.items())},
        "generated_tokens": {
            "mean": sum(toks) / len(toks),
            "median": toks[len(toks) // 2],
            "max": toks[-1],
        },  # fmt: skip
    }


def summarize_samples(rows: list[dict], k: int) -> dict:
    """Empirical sampled pass@1 (mean over samples) and pass@k (any of k correct per prompt)."""
    by_prompt: dict[str, list[bool]] = defaultdict(list)
    for r in rows:
        by_prompt[r["prompt_id"]].append(r["status"] == "correct")
    if any(len(v) != k for v in by_prompt.values()):
        raise ValueError(f"every prompt needs exactly {k} samples")
    n_samples = sum(len(v) for v in by_prompt.values())
    any_correct = sum(any(v) for v in by_prompt.values())
    return {
        "samples_per_prompt": k,
        "sampled_pass@1": Rate(sum(sum(v) for v in by_prompt.values()), n_samples).as_dict(),
        f"pass@{k}": Rate(any_correct, len(by_prompt)).as_dict(),
    }


def paired_bootstrap(
    a: dict[str, bool], b: dict[str, bool], resamples: int = 10_000, seed: int = 0
) -> dict:
    """Difference in accuracy (b - a) over the same prompts, resampling prompt ids."""
    if set(a) != set(b):
        raise ValueError("paired comparison needs identical prompt sets")
    ids = sorted(a)
    diffs = [int(b[i]) - int(a[i]) for i in ids]
    n = len(ids)
    observed = sum(diffs) / n
    rng = random.Random(seed)
    stats = sorted(sum(diffs[rng.randrange(n)] for _ in range(n)) / n for _ in range(resamples))
    return {
        "prompts": n,
        "accuracy_a": sum(a.values()) / n,
        "accuracy_b": sum(b.values()) / n,
        "difference_b_minus_a": observed,
        "bootstrap95": [stats[int(0.025 * resamples)], stats[int(0.975 * resamples) - 1]],
        "resamples": resamples,
        "bootstrap_seed": seed,
        "method": "paired bootstrap over prompt ids",
    }
