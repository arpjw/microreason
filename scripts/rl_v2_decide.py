"""Pre-declared decision rules for protocol rl-v2 and the confirmatory test evaluation.

    select  --candidate NAME=SUMMARY ... --out FILE
        Picks the GRPO v2 run with the highest val greedy show-work exact accuracy
        (reports/rl_protocol_v2.md). Ties go to the lower learning rate, i.e. the candidate
        listed first. Prints the chosen name.

    h4      --vs-sft COMPARE --vs-control COMPARE --out FILE
        Applies the spec §7 H4 rule to the two primary held-out-template comparisons
        (reports/confirmatory_plan_v1.md): v4-grpo-r2 must beat v2-sft and v2-extra-sft-r2 by
        at least 2 percentage points each, with each paired 95% interval excluding zero.

Both write a JSON record of the inputs and the outcome. They read only files the session wrote.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

H4_MIN_DIFFERENCE = 0.02


def _select(args) -> int:
    rows = []
    for item in args.candidate:
        name, path = item.split("=", 1)
        s = json.loads(Path(path).read_text())
        if s["suite"] != "val" or s["prompt_format"] != "sft-show-work-v1":
            sys.exit(f"{path}: selection uses val sft-show-work-v1 summaries only")
        acc = s["greedy"]["exact_accuracy"]
        rows.append({"name": name, "summary": path, "checkpoint": s["checkpoint"],
                     "val_exact_accuracy": acc["value"], "numerator": acc["numerator"],
                     "denominator": acc["denominator"]})  # fmt: skip
    best = rows[0]
    for r in rows[1:]:
        if r["numerator"] > best["numerator"]:  # strictly better; ties keep the earlier (lower LR)
            best = r
    out = {"rule": "highest val greedy show-work exact accuracy; ties -> lower LR (listed first)",
           "candidates": rows, "selected": best["name"],
           "selected_checkpoint": best["checkpoint"]}  # fmt: skip
    Path(args.out).write_text(json.dumps(out, indent=2) + "\n")
    print(best["name"])
    return 0


def _check(path: str, a: str, b: str) -> dict:
    c = json.loads(Path(path).read_text())
    if c["suite"] != "test_heldout_template" or c["prompt_format"] != "sft-show-work-v1":
        sys.exit(f"{path}: H4 uses test_heldout_template sft-show-work-v1 comparisons only")
    if not (c["a"].rstrip("/").endswith(a) and c["b"].rstrip("/").endswith(b)):
        sys.exit(f"{path}: expected a={a} b={b}, got a={c['a']} b={c['b']}")
    d, (lo, hi) = c["difference_b_minus_a"], c["bootstrap95"]
    return {"compare": path, "a": c["a"], "b": c["b"], "accuracy_a": c["accuracy_a"],
            "accuracy_b": c["accuracy_b"], "difference_b_minus_a": d, "bootstrap95": [lo, hi],
            "prompts": c["prompts"], "resamples": c["resamples"],
            "bootstrap_seed": c["bootstrap_seed"],
            "meets_threshold": d >= H4_MIN_DIFFERENCE, "interval_excludes_zero": lo > 0,
            "passes": d >= H4_MIN_DIFFERENCE and lo > 0}  # fmt: skip


def _h4(args) -> int:
    sft = _check(args.vs_sft, "v2-sft_test_heldout_template_sft-show-work-v1",
                 "v4-grpo-r2_test_heldout_template_sft-show-work-v1")  # fmt: skip
    ctl = _check(args.vs_control, "v2-extra-sft-r2_test_heldout_template_sft-show-work-v1",
                 "v4-grpo-r2_test_heldout_template_sft-show-work-v1")  # fmt: skip
    supported = sft["passes"] and ctl["passes"]
    out = {
        "hypothesis": "H4: exact-outcome GRPO improves held-out correctness beyond SFT and "
        "equal-cost continued SFT",
        "rule": f"both differences >= {H4_MIN_DIFFERENCE} and both paired 95% intervals > 0",
        "v4_grpo_r2_vs_v2_sft": sft,
        "v4_grpo_r2_vs_v2_extra_sft_r2": ctl,
        "h4_supported": supported,
    }
    Path(args.out).write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps({"h4_supported": supported}))
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("select")
    s.add_argument("--candidate", action="append", required=True)
    s.add_argument("--out", required=True)
    h = sub.add_parser("h4")
    h.add_argument("--vs-sft", required=True)
    h.add_argument("--vs-control", required=True)
    h.add_argument("--out", required=True)
    args = p.parse_args()
    return _select(args) if args.cmd == "select" else _h4(args)


if __name__ == "__main__":
    raise SystemExit(main())
