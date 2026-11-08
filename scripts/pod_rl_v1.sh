#!/usr/bin/env bash
# Run ON THE POD after `SETUP_PROFILE=rl scripts/pod_setup.sh` and after the v2-sft checkpoint
# has been copied to artifacts/p2-v2-sft. The RL-stage branches (reports/rl_protocol_v1.md):
#   v2-sft re-eval -> reward-variance gate -> v4-grpo -> v2-extra-sft (time-matched to GRPO)
#   -> v3-dpo, each evaluated on val right after training, then paired comparisons.
# All runs are guarded paid runs under configs/budget_rl_v1.yaml. Test suites stay sealed.
set -euo pipefail
cd "${MICROREASON_ROOT:-/root/microreason}"
# shellcheck source=scripts/pod_env.sh
source scripts/pod_env.sh

BUDGET="${BUDGET:-configs/budget_rl_v1.yaml}"
DEVICE="${DEVICE:-cuda}"
RUNS="${RUNS:-runs/rl}"
REPORTS="${REPORTS:-reports/rl_v1}"
PARENT="${PARENT:-artifacts/p2-v2-sft}"
EVAL_BATCH="${EVAL_BATCH:-128}"
SAMPLES="${SAMPLES:-4}"

: "${MICROREASON_BILLING_START:?set MICROREASON_BILLING_START}"
: "${RUNPOD_POD_ID:?RUNPOD_POD_ID is not set}"
if ! { [[ -f /root/watchdog.pid ]] && kill -0 "$(cat /root/watchdog.pid)"; }; then
  echo "error: watchdog is not armed" >&2
  exit 1
fi
paid=(--budget "$BUDGET" --pod-id "$RUNPOD_POD_ID" --billing-start "$MICROREASON_BILLING_START")
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
    --batch-size "$EVAL_BATCH" --out "$RUNS/eval/${name}_val_${fmt}" "$@" \
    > "$REPORTS/${name}_val_${fmt}_summary.json"
  echo "$name $fmt evaluated"
}

eval_branch() {  # name checkpoint
  run_eval "$1" "$2" sft-show-work-v1 --samples "$SAMPLES"
  run_eval "$1" "$2" sft-direct-v1
}

compare() {  # a b
  for fmt in sft-show-work-v1 sft-direct-v1; do
    uv run microreason compare --a "$RUNS/eval/${1}_val_${fmt}" --b "$RUNS/eval/${2}_val_${fmt}" \
      | tee "$REPORTS/${2}_vs_${1}_val_${fmt}_compare.json"
  done
}

echo "== quick CUDA checks"
timeout 600 uv run pytest -p no:cacheprovider -rN tests/test_model.py tests/test_rl.py \
  2>&1 | tail -1 | tee "$REPORTS/cuda_checks.txt"

echo "== parent checkpoint"
uv run python -c "from microreason.checkpoint import verify_checkpoint as v; \
print('v2-sft step', v('$PARENT')['step'])"

echo "== v2-sft re-evaluation (same pod and code as the branches)"
eval_branch v2-sft "$PARENT"

echo "== reward-variance gate (512 gate-partition prompts)"
set +e
uv run microreason rl gate --config configs/grpo.yaml --checkpoint "$PARENT" --device "$DEVICE" \
  --out "$REPORTS/gate.json"
gate=$?
set -e
if [[ $gate -ne 0 ]]; then
  echo "GATE FAILED (exit $gate): GRPO not run, per the declared rule"
  echo "RL_V1 STOPPED_AT_GATE"
  exit 0
fi

echo "== v4-grpo"
uv run microreason rl grpo --config configs/grpo.yaml --parent "$PARENT" --device "$DEVICE" \
  --run-dir "$RUNS/v4-grpo" "${paid[@]}" | tee "$REPORTS/v4-grpo_result.json"
g="$(final_checkpoint "$RUNS/v4-grpo")"; echo "v4-grpo: $g"
eval_branch v4-grpo "$g"
compare v2-sft v4-grpo

echo "== v2-extra-sft (time-matched to GRPO's training loop)"
wall="$(uv run python -c "import json; print(json.load(open('$RUNS/v4-grpo/run_manifest.json'))['timing']['train_wall_seconds'])")"
echo "matching GRPO train_wall_seconds = $wall"
uv run microreason train --config configs/extra_sft.yaml --parent "$PARENT" --device "$DEVICE" \
  --run-dir "$RUNS/v2-extra-sft" --max-wall-seconds "$wall" "${paid[@]}"
x="$(final_checkpoint "$RUNS/v2-extra-sft")"; echo "v2-extra-sft: $x"
eval_branch v2-extra-sft "$x"
compare v2-sft v2-extra-sft
compare v2-extra-sft v4-grpo

echo "== v3-dpo"
uv run microreason rl dpo --config configs/dpo.yaml --parent "$PARENT" --device "$DEVICE" \
  --run-dir "$RUNS/v3-dpo" "${paid[@]}" | tee "$REPORTS/v3-dpo_result.json"
d="$(final_checkpoint "$RUNS/v3-dpo")"; echo "v3-dpo: $d"
eval_branch v3-dpo "$d"
compare v2-sft v3-dpo

printf '%s\n%s\n%s\n' "$g" "$x" "$d" > "$REPORTS/final_checkpoints.txt"
echo "RL_V1 COMPLETE"
