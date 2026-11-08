#!/usr/bin/env bash
# Run ON THE POD after pod_setup.sh. The Phase 5 pilot: tests, smoke + resume on CUDA, a
# micro-batch sweep and a short full-model run, then a summary report.
#
# Requires MICROREASON_BILLING_START (ISO time the pod was created) and RUNPOD_POD_ID, and an
# approved configs/pilot_43m.yaml (approval fields filled in). Every full-model run is a paid run
# under the budget guard; the pod watchdog must already be armed.
set -euo pipefail
cd /root/microreason
export PATH="$HOME/.local/bin:$PATH"
: "${MICROREASON_BILLING_START:?set MICROREASON_BILLING_START}"
: "${RUNPOD_POD_ID:?RUNPOD_POD_ID is not set}"
if ! { [[ -f /root/watchdog.pid ]] && kill -0 "$(cat /root/watchdog.pid)"; }; then
  echo "error: watchdog is not armed" >&2
  exit 1
fi
uv run microreason config validate --config configs/pilot_43m.yaml >/dev/null \
  || { echo "error: pilot config is not approved/valid" >&2; exit 1; }

mkdir -p reports/pilot
echo "== tests on CUDA"
uv run pytest -q -p no:cacheprovider -rN 2>&1 | tail -3 | tee reports/pilot/tests.txt

echo "== smoke run and resume on CUDA"
uv run microreason train --config configs/smoke.yaml --device cuda --run-dir runs/pod_smoke_a
uv run microreason train --config configs/smoke.yaml --device cuda --run-dir runs/pod_smoke_b \
  --resume runs/pod_smoke_a/checkpoints/step_0000100

echo "== full model: micro-batch sweep (15 steps each)"
for mb in 16 32 64; do
  uv run microreason train --config configs/pilot_43m.yaml --device cuda \
    --run-dir "runs/pilot_sweep_mb$mb" --micro-batch-size "$mb" --max-steps 15 \
    || echo "micro-batch $mb failed (likely out of memory); continuing"
done
best="$(uv run python scripts/summarize_pilot.py --best-micro-batch)"
echo "fastest micro-batch that fit: $best"

echo "== full model: 200-step pilot run"
uv run microreason train --config configs/pilot_43m.yaml --device cuda \
  --run-dir runs/pilot_main --micro-batch-size "$best"

uv run python scripts/summarize_pilot.py --report reports/pilot/pilot_v1.json
echo "PILOT COMPLETE"
