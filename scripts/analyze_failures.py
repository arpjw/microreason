"""Post-hoc failure analysis of the confirmatory test evaluations (descriptive only).

Reads per-prompt eval rows (greedy, show-work) and the frozen suite records, aligns each model
trace with the gold `solution_steps`, and classifies where it first goes wrong:

  skeleton    the step's operation word differs from gold (e.g. "Multiply" where gold "Divides"),
              or the trace has a different number of steps with no earlier divergence
  copy        same operation, but the step's left-hand side (its operands) differs from gold.
              For step 1 these operands come straight from the prompt, so this is a copy error.
              Subtype `prefix`: every wrong operand is a leading-digit prefix of the gold one
              (e.g. 167 -> 16), the signature of reading only the first token of a number
  arithmetic  same operation and operands as gold, wrong result
  final       every step matches gold but the answer block does not
  correct     the verifier scored it correct

It also measures how the BPE tokenizer splits prompt numbers in each suite, and, for held-out
templates, which training template's step skeleton the model reproduced.

    uv run python scripts/analyze_failures.py --rows-dir runs/pod_results/<pod>/runs/rl2/eval \
        --out reports/failure_analysis_v1.json

Test failures are used here only to describe the models. Nothing from a test suite may enter
training data (spec §4), and any intervention this motivates needs a new frozen held-out suite.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

STEP_RE = re.compile(r"^\s*\d+\.\s*(.+?)\s*$")
NUM_RE = re.compile(r"-?\d+(?:/\d+)?")
SUITES = ("test_id", "test_extrapolation", "test_heldout_template")
MODELS = ("v2-sft", "v4-grpo-r2", "v2-extra-sft-r2")


def parse_steps(completion: str) -> list[str]:
    body = completion.split("[steps]", 1)[-1].split("[/steps]", 1)[0]
    out = []
    for line in body.splitlines():
        m = STEP_RE.match(line)
        if m:
            out.append(m.group(1))
    return out


def split_step(step: str) -> tuple[str, str, str]:
    """'Add: 57 + 47 = 104' -> ('Add', '57 + 47', '104'). Equation steps keep their label."""
    label, _, rest = step.partition(":")
    lhs, _, rhs = rest.rpartition("=")
    return label.strip(), lhs.strip(), rhs.strip()


def op_word(label: str) -> str:
    return label.split()[0] if label else ""


def classify(row: dict, rec: dict) -> dict:
    if row["status"] == "correct":
        return {"kind": "correct"}
    model = parse_steps(row["raw_completion"])
    gold = rec["solution_steps"]
    for i, g in enumerate(gold):
        if i >= len(model):
            return {"kind": "skeleton", "step": i, "detail": "too_few_steps"}
        gl, glhs, grhs = split_step(g)
        ml, mlhs, mrhs = split_step(model[i])
        if op_word(gl) != op_word(ml):
            return {"kind": "skeleton", "step": i, "detail": f"{op_word(gl)}->{op_word(ml)}"}
        if glhs != mlhs or (gl != ml):  # equation labels carry operands ("Subtract 7 from ...")
            gn, mn = NUM_RE.findall(gl + " " + glhs), NUM_RE.findall(ml + " " + mlhs)
            wrong = [(a, b) for a, b in zip(gn, mn, strict=False) if a != b]
            prefix = (
                bool(wrong)
                and len(gn) == len(mn)
                and all(
                    b.lstrip("-") and a.lstrip("-").startswith(b.lstrip("-")) and len(b) < len(a)
                    for a, b in wrong
                )
            )
            return {"kind": "copy", "step": i, "prefix": prefix}
        if grhs != mrhs:
            return {"kind": "arithmetic", "step": i}
    if len(model) != len(gold):
        return {"kind": "skeleton", "step": len(gold), "detail": "too_many_steps"}
    return {"kind": "final", "step": len(gold)}


def skeleton(steps: list[str]) -> tuple[str, ...]:
    return tuple(op_word(split_step(s)[0]) for s in steps)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--rows-dir", required=True)
    p.add_argument("--suite-dir", default="data/generated/math_v1/eval")
    p.add_argument("--train", default="data/generated/math_v1/train/train.jsonl")
    p.add_argument("--tokenizer", default="artifacts/tokenizer/v1")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    from microreason.tokenizer import MicroTokenizer

    tok = MicroTokenizer.load(args.tokenizer)
    records = {s: {} for s in SUITES}
    for s in SUITES:
        for line in open(Path(args.suite_dir) / f"{s}.jsonl"):
            r = json.loads(line)
            records[s][r["id"]] = r

    report: dict = {"note": "post-hoc, descriptive; test failures never enter training data"}

    # ---- tokenization of prompt operands per suite
    tokinfo = {}
    for s in SUITES:
        n_tok = Counter()
        for r in records[s].values():
            for v in r["params"].values():
                ids = tok.encode(f" {abs(int(v))}")
                n_tok[min(len(ids), 4)] += 1
        total = sum(n_tok.values())
        tokinfo[s] = {f"{k}_tokens": round(n_tok[k] / total, 4) for k in sorted(n_tok)}
    train_tok = Counter()
    if Path(args.train).exists():
        for i, line in enumerate(open(args.train)):
            if i >= 50_000:
                break
            for v in json.loads(line)["params"].values():
                train_tok[min(len(tok.encode(f" {abs(int(v))}")), 4)] += 1
        t = sum(train_tok.values())
        tokinfo["train_sample_50k"] = {f"{k}_tokens": round(train_tok[k] / t, 4)
                                       for k in sorted(train_tok)}  # fmt: skip
    report["operand_tokenization"] = tokinfo

    # ---- training template skeletons, from gold traces of the in-distribution test suite
    train_skel = defaultdict(Counter)
    for r in records["test_id"].values():
        train_skel[skeleton(r["solution_steps"])][r["template_id"]] += 1
    skel_to_template = {k: v.most_common(1)[0][0] for k, v in train_skel.items()}

    # ---- classify every greedy show-work row
    rows_dir = Path(args.rows_dir)
    out = {}
    for model in MODELS:
        out[model] = {}
        for s in SUITES:
            f = rows_dir / f"{model}_{s}_sft-show-work-v1" / "rows.jsonl"
            kinds = Counter()
            by_tmpl = defaultdict(Counter)
            first_step = Counter()
            prefix = Counter()
            reproduced = Counter()
            by_tok = {"all_single_token": Counter(), "some_multi_token": Counter()}
            for line in open(f):
                row = json.loads(line)
                if row["decode"] != "greedy":
                    continue
                rec = records[s][row["prompt_id"]]
                c = classify(row, rec)
                kinds[c["kind"]] += 1
                single = all(len(tok.encode(f" {abs(int(v))}")) == 1
                             for v in rec["params"].values())  # fmt: skip
                by_tok["all_single_token" if single else "some_multi_token"][c["kind"]] += 1
                by_tmpl[rec["template_id"]][c["kind"]] += 1
                if c["kind"] not in ("correct",):
                    first_step[c.get("step")] += 1
                if c["kind"] == "copy":
                    prefix["prefix" if c["prefix"] else "other"] += 1
                if s == "test_heldout_template":
                    sk = skeleton(parse_steps(row["raw_completion"]))
                    reproduced[skel_to_template.get(sk, "no_training_skeleton")] += 1
            n = sum(kinds.values())
            entry = {
                "prompts": n,
                "first_divergence": {k: round(v / n, 4) for k, v in kinds.most_common()},
                "first_divergence_step": dict(sorted(first_step.items(), key=lambda x: str(x[0]))),
                "copy_error_subtype": dict(prefix),
                "by_template": {t: dict(c.most_common()) for t, c in sorted(by_tmpl.items())},
                "by_operand_tokenization": {
                    k: dict(c.most_common()) for k, c in by_tok.items() if c
                },  # fmt: skip
            }
            if s == "test_heldout_template":
                entry["reproduced_training_skeleton"] = dict(reproduced.most_common())
                entry["gold_skeletons"] = {
                    t: list(skeleton(next(r for r in records[s].values()
                                          if r["template_id"] == t)["solution_steps"]))
                    for t in sorted(by_tmpl)
                }  # fmt: skip
            out[model][s] = entry
    report["models"] = out
    report["training_skeletons"] = {" > ".join(k): v for k, v in skel_to_template.items()}
    Path(args.out).write_text(json.dumps(report, indent=2) + "\n")

    for model in MODELS:
        for s in SUITES:
            e = out[model][s]
            print(model, s, e["first_divergence"], e["copy_error_subtype"])
    print(json.dumps(tokinfo, indent=1))
    for model in MODELS:
        rep = out[model]["test_heldout_template"]["reproduced_training_skeleton"]
        print(model, "reproduced", rep)
    print(json.dumps(out["v2-sft"]["test_heldout_template"]["gold_skeletons"], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
