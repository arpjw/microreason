"""Protocol v3 capability gate (reports/protocol_v3.md), declared before any v3 run.

Phase C (RL branches and the confirmatory test) runs only if v2-sft, on the development
held-out shape suite `val_heldout_shape` (greedy show-work), reaches both:
  exact accuracy >= 0.10   and   sampled pass@4 >= 0.20.

    python scripts/v3_capability_gate.py --summary <v2-sft val_heldout_shape show-work summary> \
        --out reports/v3_phase_b/capability_gate.json

Exit 0 if the gate passes, 3 if it does not.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

MIN_GREEDY = 0.10
MIN_PASS_AT_4 = 0.20


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--summary", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    s = json.loads(Path(a.summary).read_text())
    if s["suite"] != "val_heldout_shape" or s["prompt_format"] != "sft-show-work-v1":
        sys.exit("the gate uses the val_heldout_shape sft-show-work-v1 summary only")
    if s.get("data_version") != "v3" or not s.get("sampled"):
        sys.exit("the gate needs a data-v3 summary with sampled pass@4")
    greedy = s["greedy"]["exact_accuracy"]["value"]
    p4 = s["sampled"]["pass@4"]["value"]
    passed = greedy >= MIN_GREEDY and p4 >= MIN_PASS_AT_4
    out = {"rule": {"min_greedy_exact": MIN_GREEDY, "min_sampled_pass_at_4": MIN_PASS_AT_4},
           "greedy_exact": greedy, "sampled_pass_at_4": p4, "summary": a.summary,
           "checkpoint_id": s["checkpoint_id"], "passed": passed}  # fmt: skip
    Path(a.out).write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out))
    return 0 if passed else 3


if __name__ == "__main__":
    raise SystemExit(main())
