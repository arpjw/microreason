"""Paid-compute budget guard and ledger (spec §8).

Spend is measured from when billing started (pod creation), not from when training started, so
setup and data preparation count against the cap. Before every optimizer step the guard checks
that one more step plus the reserve stays inside the tighter of the runtime limit and the dollar
cap. After a short measurement window it also projects the cost of the remaining steps and stops
before starting work the budget cannot cover.

The ledger is written atomically after every check so the latest state survives a crash.
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

from microreason.config import BudgetConfig

STOP_CEILING = "stopped_budget_ceiling"
STOP_PROJECTION = "stopped_projected_over_budget"


class BudgetGuard:
    def __init__(
        self,
        budget: BudgetConfig,
        billing_start: float,
        pod_id: str,
        ledger_path: str | Path,
        clock: Callable[[], float] = time.time,
        projection_after_steps: int = 5,
    ):
        if not budget.paid_compute_allowed:
            raise ValueError("a budget guard is only for authorized paid runs")
        if not pod_id:
            raise ValueError("paid runs must record the pod id")
        self.budget = budget
        self.billing_start = billing_start
        self.pod_id = pod_id
        self.ledger_path = Path(ledger_path)
        self.clock = clock
        self.projection_after_steps = projection_after_steps
        self.rate = budget.hourly_rate_usd
        self.ceiling_s = budget.ceiling_hours() * 3600
        self.reserve_s = budget.reserve_minutes * 60
        if self.elapsed_s() + self.reserve_s >= self.ceiling_s:
            raise RuntimeError("budget already exhausted before training started")

    def elapsed_s(self) -> float:
        return max(0.0, self.clock() - self.billing_start)

    def accrued_usd(self) -> float:
        return self.elapsed_s() / 3600 * self.rate

    def projected_total_usd(self, step_s: float, steps_remaining: int) -> float:
        return (self.elapsed_s() + steps_remaining * step_s + self.reserve_s) / 3600 * self.rate

    def check(
        self, step_s: float | None, steps_done_this_run: int, steps_remaining: int
    ) -> str | None:
        """Return a stop reason if the next step must not start, else None."""
        reason = None
        if step_s is not None:
            if self.elapsed_s() + step_s + self.reserve_s > self.ceiling_s:
                reason = STOP_CEILING
            elif (
                steps_done_this_run >= self.projection_after_steps
                and self.projected_total_usd(step_s, steps_remaining) > self.budget.max_spend_usd
            ):
                reason = STOP_PROJECTION
        elif self.elapsed_s() + self.reserve_s > self.ceiling_s:
            reason = STOP_CEILING
        self.write(step_s, steps_remaining, reason)
        return reason

    def write(self, step_s: float | None, steps_remaining: int, stop_reason: str | None) -> dict:
        entry = {
            "provider": self.budget.provider,
            "gpu_type": self.budget.gpu_type,
            "pod_id": self.pod_id,
            "hourly_rate_usd": self.rate,
            "approved_max_usd": self.budget.max_spend_usd,
            "approved_max_runtime_hours": self.budget.max_runtime_hours,
            "approved_by": self.budget.approved_by,
            "approved_at": self.budget.approved_at,
            "shutdown_rule": self.budget.shutdown_rule,
            "billing_start": datetime.fromtimestamp(self.billing_start, UTC).isoformat(),
            "ceiling_at": datetime.fromtimestamp(
                self.billing_start + self.ceiling_s, UTC
            ).isoformat(),
            "reserve_minutes": self.budget.reserve_minutes,
            "elapsed_hours": round(self.elapsed_s() / 3600, 4),
            "estimated_accrued_usd": round(self.accrued_usd(), 4),
            "measured_step_seconds": None if step_s is None else round(step_s, 4),
            "steps_remaining": steps_remaining,
            "projected_total_usd": None
            if step_s is None
            else round(self.projected_total_usd(step_s, steps_remaining), 4),
            "stop_reason": stop_reason,
            "note": "GPU time at the listed rate only; storage and other charges are excluded.",
            "updated_at": datetime.fromtimestamp(self.clock(), UTC).isoformat(),
        }
        self.ledger_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.ledger_path.with_name(self.ledger_path.name + ".tmp")
        tmp.write_text(json.dumps(entry, indent=2, sort_keys=True) + "\n")
        os.replace(tmp, self.ledger_path)
        return entry
