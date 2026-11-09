"""Pre-declared decision rules for the data-v3 confirmatory test (reports/confirmatory_plan_v2.md).

    python scripts/v3_h4_decision.py --reports reports/v3_phase_c \
        --out reports/v3_phase_c/h4_decision.json

Reads the paired comparisons the Phase C session wrote, named
`<b>_vs_<a>_<suite>_<format>_compare.json`, and applies the rules:

- **H4** (`test_heldout_shape`, show-work): all four of v4-grpo-s{1,2} − v2-sft and
  v4-grpo-s{k} − v2-extra-sft-s{k} must be at least +2.0 points, each with a paired 95% interval
  above 0.
- **H1** (`test_id`, plain-v1): v1-math − v0-base must have its interval above 0.
- **H2** (`test_id`, SFT show-work prompts): v2-sft − v1-math must have its interval above 0.

A missing comparison leaves that hypothesis undecided; it never counts as a pass.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

H4_MIN = 0.02


def _cmp(reports: Path, a: str, b: str, suite: str, fmt: str) -> dict | None:
    f = reports / f"{b}_vs_{a}_{suite}_{fmt}_compare.json"
    if not f.exists():
        return None
    c = json.loads(f.read_text())
    if c["suite"] != suite or c["prompt_format"] != fmt:
        raise SystemExit(f"{f}: expected {suite}/{fmt}, got {c['suite']}/{c['prompt_format']}")
    if not (c["a"].rstrip("/").endswith(f"{a}_{suite}_{fmt}")
            and c["b"].rstrip("/").endswith(f"{b}_{suite}_{fmt}")):  # fmt: skip
        raise SystemExit(f"{f}: a/b do not match {a}/{b}")
    lo, hi = c["bootstrap95"]
    return {"file": str(f), "a": a, "b": b, "accuracy_a": c["accuracy_a"],
            "accuracy_b": c["accuracy_b"], "difference_b_minus_a": c["difference_b_minus_a"],
            "bootstrap95": [lo, hi], "prompts": c["prompts"], "resamples": c["resamples"],
            "bootstrap_seed": c["bootstrap_seed"], "interval_above_zero": lo > 0}  # fmt: skip


def _status(results: list[dict | None], need_min: float | None) -> str:
    if any(r is None for r in results):
        return "undecided"

    def passes(r: dict) -> bool:
        big_enough = need_min is None or r["difference_b_minus_a"] >= need_min
        return r["interval_above_zero"] and big_enough

    ok = all(passes(r) for r in results)
    return "supported" if ok else "not_supported"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--reports", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    r = Path(a.reports)
    hs, sw = "test_heldout_shape", "sft-show-work-v1"
    h4 = {
        "v4-grpo-s1_vs_v2-sft": _cmp(r, "v2-sft", "v4-grpo-s1", hs, sw),
        "v4-grpo-s1_vs_v2-extra-sft-s1": _cmp(r, "v2-extra-sft-s1", "v4-grpo-s1", hs, sw),
        "v4-grpo-s2_vs_v2-sft": _cmp(r, "v2-sft", "v4-grpo-s2", hs, sw),
        "v4-grpo-s2_vs_v2-extra-sft-s2": _cmp(r, "v2-extra-sft-s2", "v4-grpo-s2", hs, sw),
    }
    h1 = _cmp(r, "v0-base", "v1-math", "test_id", "plain-v1")
    h2 = _cmp(r, "v1-math", "v2-sft", "test_id", sw)
    out = {
        "plan": "reports/confirmatory_plan_v2.md",
        "H4": {"rule": f"all four differences >= {H4_MIN} and each interval above 0",
               "comparisons": h4, "status": _status(list(h4.values()), H4_MIN)},
        "H1": {"rule": "interval above 0", "comparison": h1, "status": _status([h1], None)},
        "H2": {"rule": "interval above 0", "comparison": h2, "status": _status([h2], None)},
    }  # fmt: skip
    Path(a.out).write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps({k: out[k]["status"] for k in ("H1", "H2", "H4")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
