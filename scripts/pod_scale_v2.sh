#!/usr/bin/env bash
# Run ON THE POD after `SETUP_PROFILE=scale scripts/pod_setup.sh`: the protocol-v2 scaled redo.
#   v0-base (1 B tokens) -> v1-math (200 M, curriculum) -> plain-v1 val evals + H1 compare
#   -> v2-sft (unchanged SFT recipe) -> SFT-format val evals of v2-sft and v1-math + H2 compares.
# All runs are guarded paid runs under configs/budget_scale_v2.yaml. Test suites stay sealed.
# Environment overrides exist only for the local dry run (DRY_RUN=1, refused on a pod).
set -euo pipefail
cd "${MICROREASON_ROOT:-/root/microreason}"
# shellcheck source=scripts/pod_env.sh
source scripts/pod_env.sh

PRETRAIN_CONFIG="${PRETRAIN_CONFIG:-configs/pretrain_v2.yaml}"
MIDTRAIN_CONFIG="${MIDTRAIN_CONFIG:-configs/midtrain_v2.yaml}"
SFT_CONFIG="${SFT_CONFIG:-configs/sft.yaml}"
BUDGET="${BUDGET:-configs/budget_scale_v2.yaml}"
DEVICE="${DEVICE:-cuda}"
RUNS="${RUNS:-runs/p2}"
REPORTS="${REPORTS:-reports/scale_v2}"
EVAL_BATCH="${EVAL_BATCH:-128}"
EVAL_LIMIT="${EVAL_LIMIT:-}"
SAMPLES="${SAMPLES:-4}"
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

final_checkpoint() {
  uv run python - "$1" <<'PY'
import json, pathlib, sys
d = pathlib.Path(sys.argv[1])
m = json.loads((d / "run_manifest.json").read_text())
if m["status"] != "completed":
    sys.exit(f"{d}: status {m['status']}; stopping the session")
print(json.loads((d / "latest.json").read_text())["path"])
PY
}

run_eval() {  # name checkpoint format [extra args]
  local name="$1" ckpt="$2" fmt="$3"; shift 3
  uv run microreason eval --suite val --checkpoint "$ckpt" --format "$fmt" --device "$DEVICE" \
    --batch-size "$EVAL_BATCH" --out "$RUNS/eval/${name}_val_${fmt}" ${limit[@]+"${limit[@]}"} "$@" \
    > "$REPORTS/${name}_val_${fmt}_summary.json"
  echo "$name $fmt evaluated"
}

compare() {  # tag a b fmt
  uv run microreason compare --a "$RUNS/eval/${2}_val_${4}" --b "$RUNS/eval/${3}_val_${4}" \
    | tee "$REPORTS/${1}_val_${4}_compare.json"
}

if [[ "$DRY_RUN" != 1 ]]; then
  echo "== quick CUDA checks"
  timeout 600 uv run pytest -p no:cacheprovider -rN tests/test_model.py tests/test_training.py \
    tests/test_sft.py 2>&1 | tail -1 | tee "$REPORTS/cuda_checks.txt"
fi

echo "== v0-base pretraining (protocol v2)"
uv run microreason train --config "$PRETRAIN_CONFIG" --device "$DEVICE" \
  --run-dir "$RUNS/v0-base" ${paid[@]+"${paid[@]}"}
v0="$(final_checkpoint "$RUNS/v0-base")"; echo "v0-base: $v0"

echo "== v1-math mid-training (protocol v2)"
uv run microreason train --config "$MIDTRAIN_CONFIG" --device "$DEVICE" --parent "$v0" \
  --run-dir "$RUNS/v1-math" ${paid[@]+"${paid[@]}"}
v1="$(final_checkpoint "$RUNS/v1-math")"; echo "v1-math: $v1"

echo "== plain-v1 val evaluation + H1"
run_eval v0-base "$v0" plain-v1
run_eval v1-math "$v1" plain-v1
compare h1 v0-base v1-math plain-v1

echo "== v2-sft supervised fine-tuning (protocol v2 parent)"
uv run microreason train --config "$SFT_CONFIG" --device "$DEVICE" --parent "$v1" \
  --run-dir "$RUNS/v2-sft" ${paid[@]+"${paid[@]}"}
v2="$(final_checkpoint "$RUNS/v2-sft")"; echo "v2-sft: $v2"

echo "== SFT-format val evaluation + H2"
run_eval v2-sft "$v2" sft-show-work-v1 --samples "$SAMPLES"
run_eval v2-sft "$v2" sft-direct-v1
run_eval v1-math "$v1" sft-show-work-v1
run_eval v1-math "$v1" sft-direct-v1
compare h2 v1-math v2-sft sft-show-work-v1
compare h2 v1-math v2-sft sft-direct-v1

printf '%s\n%s\n%s\n' "$v0" "$v1" "$v2" > "$REPORTS/final_checkpoints.txt"
echo "SCALE_V2 COMPLETE"
