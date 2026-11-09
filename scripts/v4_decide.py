"""Protocol v4 pre-declared decision rules.

See reports/protocol_v4.md and reports/confirmatory_plan_v3.md.

Every rule here was committed before the data it reads existed. Subcommands:

    plateau   v2-sft-long's held-out-val curve -> plateau report (descriptive; never blocks)
    headroom  parent sampled pass@16 - pass@1 on held-out-val >= 0.05 (exit 5 if not)
    select    pick the pilot LR with the highest held-out-val greedy accuracy (ties: lower LR)
    h4        the confirmatory H4 rule on test_heldout_shape (+ H1/H2 on test_id, secondaries)

Eval directories are `<eval-dir>/<name>_<suite>_<format>/{rows.jsonl,summary.json}` as written by
`microreason eval`. Accuracy is exact-answer greedy accuracy; every prompt is in the denominator.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

SW = "sft-show-work-v1"
H4_MIN = 0.02
HEADROOM_MIN = 0.05
PLATEAU_MAX_GAIN = 0.01
RESAMPLES = 10_000
BOOT_SEED = 0


def _summary(eval_dir: Path, name: str, suite: str, fmt: str = SW) -> dict:
    return json.loads((eval_dir / f"{name}_{suite}_{fmt}" / "summary.json").read_text())


def _rows(eval_dir: Path, name: str, suite: str, fmt: str = SW) -> dict[str, tuple[str, bool]]:
    """prompt_id -> (cluster = shape/template id, correct) for greedy rows."""
    out = {}
    for line in (eval_dir / f"{name}_{suite}_{fmt}" / "rows.jsonl").read_text().splitlines():
        r = json.loads(line)
        if r.get("decode", "greedy") != "greedy":
            continue
        out[r["prompt_id"]] = (r["template_id"], r["status"] == "correct")
    return out


def _acc(rows: dict) -> float:
    return sum(c for _, c in rows.values()) / len(rows)


def cluster_bootstrap(pairs: list[tuple[dict, dict]], resamples: int = RESAMPLES,
                      seed: int = BOOT_SEED) -> dict:  # fmt: skip
    """Mean over `pairs` (one (a, b) per training seed) of acc(b) - acc(a), with a 95% percentile
    interval from resampling shape clusters (all prompts of a shape together) with replacement.
    Every pair must cover the same prompts; seeds are held fixed (they are the declared runs)."""
    ids = sorted(pairs[0][0])
    for a, b in pairs:
        if set(a) != set(ids) or set(b) != set(ids):
            raise ValueError("every compared run must cover exactly the same prompts")
    clusters: dict[str, list[int]] = defaultdict(list)
    for i, pid in enumerate(ids):
        clusters[pairs[0][0][pid][0]].append(i)
    keys = sorted(clusters)
    # per-prompt difference averaged over seeds
    d = [sum(int(b[pid][1]) - int(a[pid][1]) for a, b in pairs) / len(pairs) for pid in ids]
    csum = {k: sum(d[i] for i in clusters[k]) for k in keys}
    csize = {k: len(clusters[k]) for k in keys}
    point = sum(d) / len(d)
    rng = random.Random(seed)
    stats = []
    for _ in range(resamples):
        pick = [keys[rng.randrange(len(keys))] for _ in keys]
        stats.append(sum(csum[k] for k in pick) / sum(csize[k] for k in pick))
    stats.sort()
    lo, hi = stats[int(0.025 * resamples)], stats[int(0.975 * resamples) - 1]
    return {"difference": point, "bootstrap95": [lo, hi], "clusters": len(keys),
            "prompts": len(ids), "seeds": len(pairs), "resamples": resamples,
            "bootstrap_seed": seed, "per_seed": [_acc(b) - _acc(a) for a, b in pairs],
            "interval_above_zero": lo > 0}  # fmt: skip


# ----------------------------------------------------------------------------- subcommands


def cmd_plateau(a) -> int:
    ev = Path(a.eval_dir)
    pts = []
    for spec in a.points:  # "examples:name"
        n, name = spec.split(":", 1)
        s = _summary(ev, name, "val_heldout_shape")
        pts.append({"examples": int(n), "name": name,
                    "accuracy": s["greedy"]["exact_accuracy"]["value"]})  # fmt: skip
    pts.sort(key=lambda p: p["examples"])
    last, prev = pts[-1], pts[-2]
    gain = last["accuracy"] - prev["accuracy"]
    out = {"rule": f"plateaued iff accuracy(final) - accuracy(half) < {PLATEAU_MAX_GAIN} "
                   "(held-out-val greedy show-work); descriptive, never blocks",
           "points": pts, "last_doubling_gain": gain,
           "plateaued": gain < PLATEAU_MAX_GAIN}  # fmt: skip
    Path(a.out).write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps({"plateaued": out["plateaued"], "last_doubling_gain": round(gain, 4)}))
    return 0


def cmd_headroom(a) -> int:
    s = _summary(Path(a.eval_dir), a.name, "val_heldout_shape")
    samp = s["sampled"]
    k = samp["samples_per_prompt"]
    gap = samp[f"pass@{k}"]["value"] - samp["sampled_pass@1"]["value"]
    temp = s["decode"]["sampled"]["temperature"]
    ok = k == 16 and temp == 1.0 and gap >= HEADROOM_MIN
    out = {"rule": f"pass@16 - sampled pass@1 >= {HEADROOM_MIN} on val_heldout_shape at "
                   "temperature 1.0", "samples": k, "temperature": temp,
           "pass_at_k": samp[f"pass@{k}"]["value"], "pass_at_1": samp["sampled_pass@1"]["value"],
           "gap": gap, "passed": ok,
           "greedy_accuracy": s["greedy"]["exact_accuracy"]["value"]}  # fmt: skip
    Path(a.out).write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps({"passed": ok, "gap": round(gap, 4)}))
    return 0 if ok else 5


def cmd_select(a) -> int:
    ev = Path(a.eval_dir)
    cands = []
    for spec in a.candidates:  # "lr:name"
        lr, name = spec.split(":", 1)
        s = _summary(ev, name, "val_heldout_shape")
        cands.append({"lr": float(lr), "lr_text": lr, "name": name,
                      "accuracy": s["greedy"]["exact_accuracy"]["value"],
                      "correct": s["greedy"]["exact_accuracy"]["numerator"]})  # fmt: skip
    best = sorted(cands, key=lambda c: (-c["correct"], c["lr"]))[0]
    out = {"rule": "highest held-out-val greedy show-work accuracy; ties -> lower LR",
           "arm": a.arm, "candidates": cands, "selected": best}  # fmt: skip
    if a.baseline:
        out["baseline"] = {"name": a.baseline, "accuracy": _summary(
            ev, a.baseline, "val_heldout_shape")["greedy"]["exact_accuracy"]["value"]}  # fmt: skip
    Path(a.out).write_text(json.dumps(out, indent=2) + "\n")
    print(best["lr_text"])  # exactly as given, so callers can rebuild run names
    return 0


def cmd_h4(a) -> int:
    ev = Path(a.eval_dir)
    hs = "test_heldout_shape"
    seeds = [1, 2, 3]

    def have(name, suite, fmt=SW):
        return (ev / f"{name}_{suite}_{fmt}" / "rows.jsonl").exists()

    out: dict = {"plan": "reports/confirmatory_plan_v3.md"}
    need = (
        ["v2-sft-long"] + [f"v4-grpo-s{s}" for s in seeds] + [f"v2-extra-sft-s{s}" for s in seeds]
    )
    if all(have(n, hs) for n in need):
        parent = _rows(ev, "v2-sft-long", hs)
        g = {s: _rows(ev, f"v4-grpo-s{s}", hs) for s in seeds}
        x = {s: _rows(ev, f"v2-extra-sft-s{s}", hs) for s in seeds}
        vs_parent = cluster_bootstrap([(parent, g[s]) for s in seeds])
        vs_control = cluster_bootstrap([(x[s], g[s]) for s in seeds])
        per_seed = vs_parent["per_seed"] + vs_control["per_seed"]
        ok = (vs_parent["difference"] >= H4_MIN and vs_control["difference"] >= H4_MIN
              and vs_parent["interval_above_zero"] and vs_control["interval_above_zero"]
              and all(v > 0 for v in per_seed))  # fmt: skip
        out["H4"] = {"rule": f"seed-mean difference >= {H4_MIN} and shape-cluster bootstrap 95% "
                             "interval above 0, vs v2-sft-long AND vs the time-matched "
                             "v2-extra-sft; and all 6 per-seed differences > 0",
                     "vs_parent": vs_parent, "vs_control": vs_control,
                     "status": "supported" if ok else "not_supported"}  # fmt: skip
    else:
        missing = [n for n in need if not have(n, hs)]
        out["H4"] = {"status": "undecided", "missing": missing}
    # H1 / H2 on test_id (carried over unchanged from confirmatory plan v2)
    for h, (a_, b_, fmt) in {"H1": ("v0-base", "v1-math", "plain-v1"),
                              "H2": ("v1-math", "v2-sft", SW)}.items():  # fmt: skip
        if have(a_, "test_id", fmt) and have(b_, "test_id", fmt):
            r = cluster_bootstrap([(_rows(ev, a_, "test_id", fmt), _rows(ev, b_, "test_id", fmt))])
            out[h] = {
                "rule": "shape-cluster bootstrap 95% interval above 0",
                **r,
                "status": "supported" if r["interval_above_zero"] else "not_supported",
            }
        else:
            out[h] = {"status": "undecided"}
    # pre-declared secondary (exploratory) comparisons, seed 1 only
    sec = {}
    for b_, a_ in [("v4-grpo-s1", "v4-rft-s1"), ("v4-rft-s1", "v2-sft-long"),
                   ("v4-grpo-step-s1", "v4-grpo-s1"), ("v4-grpo-step-s1", "v2-sft-long"),
                   ("v4-rft-s1", "v2-extra-sft-s1")]:  # fmt: skip
        if have(a_, hs) and have(b_, hs):
            sec[f"{b_}_vs_{a_}"] = cluster_bootstrap([(_rows(ev, a_, hs), _rows(ev, b_, hs))])
    out["secondary"] = sec
    Path(a.out).write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps({k: out[k]["status"] for k in ("H1", "H2", "H4")}))
    return 0


def cmd_replication(a) -> int:
    """reports/replication_v4x.md: on the fresh frozen v4x extrapolation suite, the seed-mean
    v4-grpo - v2-extra-sft difference must have a shape-cluster 95% interval above 0 and all
    three per-seed differences must be > 0. No magnitude threshold."""
    ev = Path(a.eval_dir)
    suite = "test_extrapolation_v4x"
    seeds = (1, 2, 3)
    need = ["v2-sft-long", *(f"v4-grpo-s{s}" for s in seeds),
            *(f"v2-extra-sft-s{s}" for s in seeds)]  # fmt: skip
    missing = [n for n in need if not (ev / f"{n}_{suite}_{SW}" / "rows.jsonl").exists()]
    out: dict = {"plan": "reports/replication_v4x.md", "suite": suite}
    if missing:
        out["status"] = "undecided"
        out["missing"] = missing
    else:
        p = _rows(ev, "v2-sft-long", suite)
        g = {s: _rows(ev, f"v4-grpo-s{s}", suite) for s in seeds}
        x = {s: _rows(ev, f"v2-extra-sft-s{s}", suite) for s in seeds}
        prim = cluster_bootstrap([(x[s], g[s]) for s in seeds])
        ok = prim["interval_above_zero"] and all(v > 0 for v in prim["per_seed"])
        out.update(rule="seed-mean grpo - control interval above 0 and all 3 seeds > 0",
                   grpo_vs_control=prim, status="replicated" if ok else "not_replicated",
                   grpo_vs_parent=cluster_bootstrap([(p, g[s]) for s in seeds]),
                   control_vs_parent=cluster_bootstrap([(p, x[s]) for s in seeds]),
                   accuracy={n: _acc(_rows(ev, n, suite)) for n in need})  # fmt: skip
        for n in ("v4-rft-s1", "v4-grpo-step-s1"):
            if (ev / f"{n}_{suite}_{SW}" / "rows.jsonl").exists():
                out["accuracy"][n] = _acc(_rows(ev, n, suite))
    Path(a.out).write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps({"status": out["status"]}))
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    pl = sub.add_parser("plateau")
    pl.add_argument("--eval-dir", required=True)
    pl.add_argument("--points", nargs="+", required=True, help="examples:name ...")
    pl.add_argument("--out", required=True)
    hr = sub.add_parser("headroom")
    hr.add_argument("--eval-dir", required=True)
    hr.add_argument("--name", required=True)
    hr.add_argument("--out", required=True)
    se = sub.add_parser("select")
    se.add_argument("--eval-dir", required=True)
    se.add_argument("--arm", required=True)
    se.add_argument("--candidates", nargs="+", required=True, help="lr:name ...")
    se.add_argument("--baseline", default=None)
    se.add_argument("--out", required=True)
    h4 = sub.add_parser("h4")
    h4.add_argument("--eval-dir", required=True)
    h4.add_argument("--out", required=True)
    rp = sub.add_parser("replication")
    rp.add_argument("--eval-dir", required=True)
    rp.add_argument("--out", required=True)
    a = p.parse_args(argv)
    return {"plateau": cmd_plateau, "headroom": cmd_headroom, "select": cmd_select,
            "h4": cmd_h4, "replication": cmd_replication}[a.cmd](a)  # fmt: skip


if __name__ == "__main__":
    sys.exit(main())
