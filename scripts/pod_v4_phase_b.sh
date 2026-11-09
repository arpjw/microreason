#!/usr/bin/env bash
# Run ON THE POD after `SETUP_PROFILE=v4 scripts/pod_setup.sh` (v3 suites regenerated against
# their frozen hashes, tokenizer v2 verified, v4 SFT pool built) and after artifacts/p3-v2-sft has
# been uploaded. Protocol v4 Phase B (reports/protocol_v4.md), held-out-VAL only:
#   RL pool v4 (background, CPU) | v2-sft-long (3 M fresh SFT examples) -> plateau curve ->
#   headroom gate -> KV-cache parity -> filter-yield gate -> GRPO LR pilot (3 LRs x 100 steps) ->
#   control LR pilot (2 LRs, time-matched) -> selection.json. Stops there; Phase C is separate.
# Every test suite stays sealed: run_eval refuses test_*.
set -euo pipefail
cd "${MICROREASON_ROOT:-/root/microreason}"
# shellcheck source=scripts/pod_env.sh
source scripts/pod_env.sh

BUDGET="${BUDGET:-configs/budget_v4_phase_b.yaml}"
DEVICE="${DEVICE:-cuda}"
RUNS="${RUNS:-runs/p4}"
REPORTS="${REPORTS:-reports/v4_phase_b}"
EVAL_BATCH="${EVAL_BATCH:-256}"
MAX_NEW="${MAX_NEW:-320}"
TOK=artifacts/tokenizer/v2
SW=sft-show-work-v1
GRPO_PILOT_LRS=(0.000001 0.000003 0.00001)
SFT_PILOT_LRS=(0.000005 0.00002)

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

train_wall() {
  uv run python -c "import json,sys; print(json.load(open(sys.argv[1]))['timing']['train_wall_seconds'])" \
    "$1/run_manifest.json"
}

run_eval() {  # name checkpoint suite format [extra args]
  local name="$1" ckpt="$2" suite="$3" fmt="$4"; shift 4
  if [[ "$suite" == test* ]]; then
    echo "error: test suites stay sealed in protocol v4 Phase B" >&2
    exit 1
  fi
  uv run microreason eval --suite "$suite" --checkpoint "$ckpt" --format "$fmt" \
    --device "$DEVICE" --batch-size "$EVAL_BATCH" --data-version v3 --kv-cache \
    --max-new-tokens "$MAX_NEW" --tokenizer "$TOK" --out "$RUNS/eval/${name}_${suite}_${fmt}" \
    "$@" > "$REPORTS/${name}_${suite}_${fmt}_summary.json"
  echo "$name $suite $fmt evaluated"
}

echo "== quick CUDA checks"
timeout 900 uv run pytest -p no:cacheprovider -rN tests/test_model.py tests/test_rl.py \
  tests/test_rl_v4.py tests/test_eval.py 2>&1 | tail -1 | tee "$REPORTS/cuda_checks.txt"

echo "== RL pool v4 (background, CPU; needed only after v2-sft-long)"
( uv run microreason data generate-rl-pool-v3 --config configs/data_v3.yaml \
    --pool configs/rl_pool_v4.yaml > "$REPORTS/rl_pool_v4.json" 2> "$REPORTS/rl_pool_v4.log" \
  && touch "$REPORTS/RL_POOL_DONE" ) &
pool_pid=$!

echo "== v2-sft-long: p3-v2-sft + 3 M fresh SFT examples (configs/sft_long_v4.yaml)"
uv run python -c "from microreason.checkpoint import verify_checkpoint as v; \
print('p3-v2-sft step', v('artifacts/p3-v2-sft')['step'])"
uv run microreason train --config configs/sft_long_v4.yaml --run-dir "$RUNS/v2-sft-long" \
  --device "$DEVICE" "${paid[@]}"
PARENT="$(final_checkpoint "$RUNS/v2-sft-long")"
echo "v2-sft-long: $PARENT"
printf '%s\n' "$PARENT" > "$REPORTS/final_checkpoints.txt"

echo "== plateau curve (held-out-val greedy at 375 k / 750 k / 1.5 M / 3 M examples)"
points=()
for step in 5860 11720 23440; do
  ck="$RUNS/v2-sft-long/checkpoints/step_$(printf '%07d' "$step")"
  ex=$(( step * 64 ))
  run_eval "sft-long-$ex" "$ck" val_heldout_shape "$SW"
  points+=("$ex:sft-long-$ex")
done
run_eval v2-sft-long "$PARENT" val_heldout_shape "$SW" --samples 16 --temperature 1.0
run_eval v2-sft-long "$PARENT" val "$SW"
run_eval v2-sft-long "$PARENT" val sft-direct-v1
points+=("3000000:v2-sft-long")
uv run python scripts/v4_decide.py plateau --eval-dir "$RUNS/eval" --points "${points[@]}" \
  --out "$REPORTS/plateau.json"

echo "== headroom gate (pass@16 - pass@1 >= 0.05 on held-out-val, temperature 1.0)"
set +e
uv run python scripts/v4_decide.py headroom --eval-dir "$RUNS/eval" --name v2-sft-long \
  --out "$REPORTS/headroom.json"
hr=$?
set -e
if [[ $hr -ne 0 ]]; then
  echo "HEADROOM GATE FAILED: no RL under protocol v4, per the declared rule"
  echo "V4_PHASE_B STOPPED_AT_HEADROOM"
  kill "$pool_pid" 2>/dev/null || true
  exit 0
fi

echo "== waiting for RL pool v4"
wait "$pool_pid"
[[ -f "$REPORTS/RL_POOL_DONE" ]] || { echo "error: RL pool build failed" >&2; exit 1; }
cat "$REPORTS/rl_pool_v4.json"

echo "== KV-cache sampler parity on v2-sft-long (512 gate-partition prompts)"
set +e
uv run microreason rl kv-check --config configs/grpo_v4_pilot.yaml --checkpoint "$PARENT" \
  --device "$DEVICE" --out "$REPORTS/kv_check.json" > /dev/null
kv=$?
set -e
if [[ $kv -ne 0 ]]; then
  echo "V4_PHASE_B STOPPED_AT_KV_CHECK"
  exit 0
fi

echo "== filter-yield gate (512 gate prompts, G=16, filter 2..14)"
set +e
uv run microreason rl gate --config configs/grpo_v4_pilot.yaml --checkpoint "$PARENT" \
  --device "$DEVICE" --out "$REPORTS/gate.json"
gate=$?
set -e
if [[ $gate -ne 0 ]]; then
  echo "V4_PHASE_B STOPPED_AT_GATE"
  exit 0
fi

echo "== GRPO LR pilot (100 steps each, pilot seed 80; held-out-val selection)"
cands=()
for lr in "${GRPO_PILOT_LRS[@]}"; do
  n="pilot-grpo-lr$lr"
  uv run microreason rl grpo --config configs/grpo_v4_pilot.yaml --parent "$PARENT" --lr "$lr" \
    --device "$DEVICE" --run-dir "$RUNS/$n" "${paid[@]}" | tee "$REPORTS/${n}_result.json"
  run_eval "$n" "$(final_checkpoint "$RUNS/$n")" val_heldout_shape "$SW"
  cands+=("$lr:$n")
done
GRPO_LR="$(uv run python scripts/v4_decide.py select --eval-dir "$RUNS/eval" --arm grpo \
  --candidates "${cands[@]}" --baseline v2-sft-long --out "$REPORTS/selection_grpo.json")"
echo "selected GRPO lr $GRPO_LR"
wall="$(train_wall "$RUNS/pilot-grpo-lr$GRPO_LR")"

echo "== control LR pilot (each time-matched to the selected GRPO pilot: $wall s)"
cands=()
for lr in "${SFT_PILOT_LRS[@]}"; do
  n="pilot-sft-lr$lr"
  uv run microreason train --config configs/extra_sft_v4_pilot.yaml --parent "$PARENT" \
    --peak-lr "$lr" --max-wall-seconds "$wall" --device "$DEVICE" --run-dir "$RUNS/$n" "${paid[@]}"
  run_eval "$n" "$(final_checkpoint "$RUNS/$n")" val_heldout_shape "$SW"
  cands+=("$lr:$n")
done
SFT_LR="$(uv run python scripts/v4_decide.py select --eval-dir "$RUNS/eval" --arm extra_sft \
  --candidates "${cands[@]}" --baseline v2-sft-long --out "$REPORTS/selection_sft.json")"
echo "selected control lr $SFT_LR"
uv run python - "$REPORTS" "$GRPO_LR" "$SFT_LR" <<'PY'
import json, sys
from pathlib import Path
r = Path(sys.argv[1])
json.dump({"grpo_lr": float(sys.argv[2]), "extra_sft_peak_lr": float(sys.argv[3]),
           "grpo": json.loads((r / "selection_grpo.json").read_text()),
           "extra_sft": json.loads((r / "selection_sft.json").read_text())},
          open(r / "selection.json", "w"), indent=2)
PY
echo "V4_PHASE_B COMPLETE"
