#!/usr/bin/env bash
# Run ON THE POD after `SETUP_PROFILE=sft scripts/pod_setup.sh` and after the v1-math checkpoint
# has been uploaded to artifacts/v1-math. The SFT session:
#   v2-sft (parent v1-math) -> val evals of v2-sft and v1-math under identical SFT prompts
#   (show-work and direct, greedy; plus sampled pass@4 for v2-sft show-work)
#   -> paired H2 comparisons (v2-sft minus v1-math), per format. Test suites stay sealed.
# Environment overrides exist only for the local dry run (DRY_RUN=1, refused on a pod).
set -euo pipefail
cd "${MICROREASON_ROOT:-/root/microreason}"
# shellcheck source=scripts/pod_env.sh
source scripts/pod_env.sh

SFT_CONFIG="${SFT_CONFIG:-configs/sft.yaml}"
PARENT="${PARENT:-artifacts/v1-math}"
BUDGET="${BUDGET:-configs/budget_stage2.yaml}"
DEVICE="${DEVICE:-cuda}"
RUNS="${RUNS:-runs}"
REPORTS="${REPORTS:-reports/stage2}"
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

echo "== parent checkpoint"
uv run python -c "from microreason.checkpoint import verify_checkpoint as v; import sys; \
m = v(sys.argv[1]); print('verified', sys.argv[1], 'step', m['step'])" "$PARENT"

if [[ "$DRY_RUN" != 1 ]]; then
  echo "== quick CUDA checks"
  timeout 600 uv run pytest -p no:cacheprovider -rN tests/test_model.py tests/test_sft.py \
    2>&1 | tail -1 | tee "$REPORTS/cuda_checks.txt"
fi

echo "== v2-sft supervised fine-tuning"
uv run microreason train --config "$SFT_CONFIG" --device "$DEVICE" --parent "$PARENT" \
  --run-dir "$RUNS/v2-sft" ${paid[@]+"${paid[@]}"}
v2="$(uv run python - "$RUNS/v2-sft" <<'PY'
import json, pathlib, sys
d = pathlib.Path(sys.argv[1])
m = json.loads((d / "run_manifest.json").read_text())
if m["status"] != "completed":
    sys.exit(f"{d}: status {m['status']}; stopping the session")
print(json.loads((d / "latest.json").read_text())["path"])
PY
)"
echo "v2-sft: $v2"

run_eval() {  # name checkpoint format [extra args]
  local name="$1" ckpt="$2" fmt="$3"; shift 3
  uv run microreason eval --suite val --checkpoint "$ckpt" --format "$fmt" --device "$DEVICE" \
    --batch-size "$EVAL_BATCH" --out "$RUNS/eval/${name}_val_${fmt}" ${limit[@]+"${limit[@]}"} "$@" \
    > "$REPORTS/${name}_val_${fmt}_summary.json"
  echo "$name $fmt evaluated"
}

echo "== val evaluation under SFT prompts"
run_eval v2-sft "$v2" sft-show-work-v1 --samples "$SAMPLES"
run_eval v2-sft "$v2" sft-direct-v1
run_eval v1-math "$PARENT" sft-show-work-v1
run_eval v1-math "$PARENT" sft-direct-v1

echo "== H2 paired comparisons on val (v2-sft minus v1-math)"
for fmt in sft-show-work-v1 sft-direct-v1; do
  uv run microreason compare --a "$RUNS/eval/v1-math_val_${fmt}" --b "$RUNS/eval/v2-sft_val_${fmt}" \
    | tee "$REPORTS/h2_val_${fmt}_compare.json"
done
printf '%s\n' "$v2" > "$REPORTS/final_checkpoints.txt"
echo "STAGE2 COMPLETE"
