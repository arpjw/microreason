"""Summarize pilot runs into throughput, memory and projected per-stage cost.

python scripts/summarize_pilot.py --best-micro-batch   # print fastest micro-batch that ran
python scripts/summarize_pilot.py --report PATH        # write the full pilot report
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Planned processed tokens per causal-LM stage (spec §6).
STAGE_TOKENS = {"pretrain (v0-base)": 120_000_000, "midtrain (v1-math)": 40_000_000}


def _manifest(run: Path) -> dict | None:
    p = run / "run_manifest.json"
    return json.loads(p.read_text()) if p.exists() else None


def _train_rows(run: Path) -> list[dict]:
    p = run / "metrics.jsonl"
    if not p.exists():
        return []
    return [r for r in map(json.loads, p.read_text().splitlines()) if r["event"] == "train"]


def run_summary(run: Path) -> dict | None:
    m = _manifest(run)
    if m is None:
        return None
    stage = m["resolved_config"]["stage"]
    rows = _train_rows(run)
    # Skip the first logged steps (CUDA warmup / allocator growth) when timing.
    steady = [r["step_seconds"] for r in rows[1:]] or [r["step_seconds"] for r in rows]
    steady.sort()
    step_s = steady[len(steady) // 2] if steady else None
    tps = stage["effective_batch_tokens"] / step_s if step_s else None
    return {
        "status": m["status"],
        "micro_batch_size": stage["micro_batch_size"],
        "steps_completed": m["steps_completed"],
        "median_step_seconds": step_s,
        "tokens_per_second": round(tps) if tps else None,
        "peak_memory_gib": max((r.get("peak_memory_gib") or 0) for r in rows) if rows else None,
        "final_eval": m.get("final_eval"),
        "error": m.get("error"),
    }


RUNS = Path("runs")


def best_micro_batch() -> int:
    best = None
    for run in sorted(RUNS.glob("pilot_sweep_mb*")):
        s = run_summary(run)
        if s and s["status"] == "completed" and s["tokens_per_second"]:
            if best is None or s["tokens_per_second"] > best[1]:
                best = (s["micro_batch_size"], s["tokens_per_second"])
    if best is None:
        raise SystemExit("no sweep run completed")
    return best[0]


def report() -> dict:
    runs = {r.name: run_summary(r) for r in sorted(RUNS.glob("p*")) if _manifest(r)}
    main = runs.get("pilot_main")
    ledger_path = RUNS / "pilot_main" / "budget_ledger.json"
    ledger = json.loads(ledger_path.read_text()) if ledger_path.exists() else None
    projections = {}
    if main and main["tokens_per_second"] and ledger:
        rate = ledger["hourly_rate_usd"]
        for name, tokens in STAGE_TOKENS.items():
            hours = tokens / main["tokens_per_second"] / 3600
            projections[name] = {
                "tokens": tokens,
                "hours_at_measured_throughput": round(hours, 3),
                "gpu_cost_usd_at_listed_rate": round(hours * rate, 2),
            }
    tests = Path("reports/pilot/tests.txt")
    return {
        "runs": runs,
        "tests": tests.read_text().strip() if tests.exists() else None,
        "budget_ledger": ledger,
        "stage_projections": projections,
        "note": "Projections exclude evaluation, checkpointing, storage and setup time; "
        "they are estimates from a 200-step run.",
    }


def main() -> int:
    global RUNS
    if "--runs-dir" in sys.argv:
        RUNS = Path(sys.argv[sys.argv.index("--runs-dir") + 1])
    if "--best-micro-batch" in sys.argv:
        print(best_micro_batch())
        return 0
    if "--report" in sys.argv:
        out = Path(sys.argv[sys.argv.index("--report") + 1])
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report(), indent=2, sort_keys=True) + "\n")
        print(f"wrote {out}")
        return 0
    print(__doc__)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
