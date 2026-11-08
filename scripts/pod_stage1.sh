#!/usr/bin/env bash
# Run ON THE POD after pod_setup.sh: the first training session.
#   v0-base pretraining -> v1-math mid-training (parent = final v0-base checkpoint)
#   -> greedy val evaluation of both (plain-v1) -> paired H1 comparison (v1 - v0).
# Test suites stay sealed. Every training run is a guarded paid run under the approved budget.
#
# Environment overrides exist only for the local dry run (DRY_RUN=1, refused on a pod).
set -euo pipefail
cd "${MICROREASON_ROOT:-/root/microreason}"
# shellcheck source=scripts/pod_env.sh
source scripts/pod_env.sh

PRETRAIN_CONFIG="${PRETRAIN_CONFIG:-configs/pretrain.yaml}"
MIDTRAIN_CONFIG="${MIDTRAIN_CONFIG:-configs/midtrain.yaml}"
BUDGET="${BUDGET:-configs/budget_stage1.yaml}"
DEVICE="${DEVICE:-cuda}"
RUNS="${RUNS:-runs}"
REPORTS="${REPORTS:-reports/stage1}"
EVAL_BATCH="${EVAL_BATCH:-128}"
EVAL_LIMIT="${EVAL_LIMIT:-}"
DRY_RUN="${DRY_RUN:-0}"

paid=()
if [[ "$DRY_RUN" == 1 ]]; then
  if [[ -n "${RUNPOD_POD_ID:-}" ]]; then
    echo "error: DRY_RUN is not allowed on a pod" >&2
    exit 1
  fi
else
  : "${MICROREASON_BILLING_START:?set MICROREASON_BILLING_START}"
  : "${RUNPOD_POD_ID:?RUNPOD_POD_ID is not set}"
  if ! { [[ -f /root/watchdog.pid ]] && kill -0 "$(cat /root/watchdog.pid)"; }; then
    echo "error: watchdog is not armed" >&2
    exit 1
  fi
  paid=(--budget "$BUDGET" --pod-id "$RUNPOD_POD_ID" --billing-start "$MICROREASON_BILLING_START")
fi
limit=()
[[ -n "$EVAL_LIMIT" ]] && limit=(--limit "$EVAL_LIMIT")
mkdir -p "$RUNS/eval" "$REPORTS"

final_checkpoint() {  # print the final checkpoint of a run, or stop the session if it did not finish
  uv run python - "$1" <<'PY'
import json, pathlib, sys
d = pathlib.Path(sys.argv[1])
m = json.loads((d / "run_manifest.json").read_text())
if m["status"] != "completed":
    sys.exit(f"{d}: status {m['status']}; stopping the session")
print(json.loads((d / "latest.json").read_text())["path"])
PY
}

if [[ "$DRY_RUN" != 1 ]]; then
  echo "== quick CUDA checks"
  timeout 600 uv run pytest -p no:cacheprovider -rN tests/test_model.py tests/test_training.py \
    2>&1 | tail -1 | tee "$REPORTS/cuda_checks.txt"
fi

echo "== v0-base pretraining"
uv run microreason train --config "$PRETRAIN_CONFIG" --device "$DEVICE" \
  --run-dir "$RUNS/v0-base" ${paid[@]+"${paid[@]}"}
v0="$(final_checkpoint "$RUNS/v0-base")"
echo "v0-base: $v0"

echo "== v1-math mid-training"
uv run microreason train --config "$MIDTRAIN_CONFIG" --device "$DEVICE" --parent "$v0" \
  --run-dir "$RUNS/v1-math" ${paid[@]+"${paid[@]}"}
v1="$(final_checkpoint "$RUNS/v1-math")"
echo "v1-math: $v1"

echo "== val evaluation (greedy, plain-v1)"
for pair in "v0-base:$v0" "v1-math:$v1"; do
  name="${pair%%:*}"; ckpt="${pair#*:}"
  uv run microreason eval --suite val --checkpoint "$ckpt" --format plain-v1 --device "$DEVICE" \
    --batch-size "$EVAL_BATCH" --out "$RUNS/eval/${name}_val_plain" ${limit[@]+"${limit[@]}"} \
    > "$REPORTS/${name}_val_plain_summary.json"
  echo "$name val evaluated"
done

echo "== H1 paired comparison on val (v1-math minus v0-base)"
uv run microreason compare --a "$RUNS/eval/v0-base_val_plain" --b "$RUNS/eval/v1-math_val_plain" \
  | tee "$REPORTS/h1_val_compare.json"
printf '%s\n%s\n' "$v0" "$v1" > "$REPORTS/final_checkpoints.txt"
echo "STAGE1 COMPLETE"
