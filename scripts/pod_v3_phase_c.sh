#!/usr/bin/env bash
# Run ON THE POD after `SETUP_PROFILE=v3c scripts/pod_setup.sh` and after the Phase B checkpoints
# have been copied to artifacts/p3-v2-sft (required), artifacts/p3-v0-base and artifacts/p3-v1-math
# (for H1/H2; marked by artifacts/P3_BASE_UPLOAD_COMPLETE). Protocol v3 Phase C
# (reports/protocol_v3.md), then the confirmatory test (reports/confirmatory_plan_v2.md):
#   KV-cache check -> reward-variance gate -> v2-sft val re-eval -> for seeds 1, 2: v4-grpo and
#   its time-matched v2-extra-sft (val + held-out-val evals) -> test suites unsealed -> primary
#   and secondary test evals -> paired comparisons -> H4 / H1 / H2 rules.
# Training runs are guarded paid runs under configs/budget_v3_phase_c.yaml.
set -euo pipefail
cd "${MICROREASON_ROOT:-/root/microreason}"
# shellcheck source=scripts/pod_env.sh
source scripts/pod_env.sh

BUDGET="${BUDGET:-configs/budget_v3_phase_c.yaml}"
DEVICE="${DEVICE:-cuda}"
RUNS="${RUNS:-runs/p3c}"
REPORTS="${REPORTS:-reports/v3_phase_c}"
PARENT="${PARENT:-artifacts/p3-v2-sft}"
V0="${V0:-artifacts/p3-v0-base}"
V1="${V1:-artifacts/p3-v1-math}"
BASE_WAIT_MINUTES="${BASE_WAIT_MINUTES:-30}"
EVAL_BATCH="${EVAL_BATCH:-128}"
SAMPLES="${SAMPLES:-4}"
MAX_NEW="${MAX_NEW:-320}"
TOK=artifacts/tokenizer/v2

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

run_eval() {  # name checkpoint suite format [extra args]
  local name="$1" ckpt="$2" suite="$3" fmt="$4"; shift 4
  local extra=()
  if [[ "$suite" == test* ]]; then
    [[ -f "$REPORTS/test_unsealed_at.txt" ]] || { echo "error: test suite before unsealing" >&2; exit 1; }
    extra=(--allow-test)
  fi
  uv run microreason eval --suite "$suite" --checkpoint "$ckpt" --format "$fmt" \
    --device "$DEVICE" --batch-size "$EVAL_BATCH" --data-version v3 --kv-cache \
    --max-new-tokens "$MAX_NEW" --tokenizer "$TOK" --out "$RUNS/eval/${name}_${suite}_${fmt}" \
    "${extra[@]}" "$@" > "$REPORTS/${name}_${suite}_${fmt}_summary.json"
  echo "$name $suite $fmt evaluated"
}

compare() {  # a b suite format   (writes <b>_vs_<a>_<suite>_<format>_compare.json)
  uv run microreason compare --a "$RUNS/eval/${1}_${3}_${4}" --b "$RUNS/eval/${2}_${3}_${4}" \
    > "$REPORTS/${2}_vs_${1}_${3}_${4}_compare.json"
  uv run python -c "import json,sys; c=json.load(open(sys.argv[1])); \
print('compared', sys.argv[2], round(c['difference_b_minus_a'],4), [round(x,4) for x in c['bootstrap95']])" \
    "$REPORTS/${2}_vs_${1}_${3}_${4}_compare.json" "${2}-${1}:${3}:${4}"
}

SW=sft-show-work-v1

echo "== quick CUDA checks"
timeout 900 uv run pytest -p no:cacheprovider -rN tests/test_model.py tests/test_rl.py \
  tests/test_eval.py tests/test_data_v3.py 2>&1 | tail -1 | tee "$REPORTS/cuda_checks.txt"

echo "== parent checkpoint and RL pool"
uv run python -c "from microreason.checkpoint import verify_checkpoint as v; \
print('p3-v2-sft step', v('$PARENT')['step'])"
uv run python -c "import json; m=json.load(open('data/manifests/math_train_rl_v3.json')); \
print('rl pool', m['record_count'], m['sha256'][:12], m['partitions'])"

echo "== KV-cache sampler parity on p3-v2-sft (512 gate-partition prompts)"
set +e
uv run microreason rl kv-check --config configs/grpo_v3_seed1.yaml --checkpoint "$PARENT" \
  --device "$DEVICE" --out "$REPORTS/kv_check.json" > /dev/null
kv=$?
set -e
if [[ $kv -ne 0 ]]; then
  echo "KV CHECK FAILED (exit $kv): no Phase C run, per the declared rule"
  echo "V3_PHASE_C STOPPED_AT_KV_CHECK"
  exit 0
fi
echo "kv check passed"

echo "== reward-variance gate (512 gate-partition prompts, group 8)"
set +e
uv run microreason rl gate --config configs/grpo_v3_seed1.yaml --checkpoint "$PARENT" \
  --device "$DEVICE" --out "$REPORTS/gate.json"
gate=$?
set -e
if [[ $gate -ne 0 ]]; then
  echo "GATE FAILED (exit $gate): GRPO not run, per the declared rule"
  echo "V3_PHASE_C STOPPED_AT_GATE"
  exit 0
fi
echo "reward-variance gate passed"

echo "== v2-sft val re-evaluation (same pod and code as the branches)"
run_eval v2-sft "$PARENT" val "$SW"
run_eval v2-sft "$PARENT" val_heldout_shape "$SW"

ckpts=("v2-sft:$PARENT")
for s in 1 2; do
  g="v4-grpo-s$s"; x="v2-extra-sft-s$s"
  echo "== $g"
  uv run microreason rl grpo --config "configs/grpo_v3_seed$s.yaml" --parent "$PARENT" \
    --device "$DEVICE" --run-dir "$RUNS/$g" "${paid[@]}" | tee "$REPORTS/${g}_result.json"
  gc="$(final_checkpoint "$RUNS/$g")"; echo "$g: $gc"
  run_eval "$g" "$gc" val "$SW"
  run_eval "$g" "$gc" val_heldout_shape "$SW"
  compare v2-sft "$g" val_heldout_shape "$SW"
  echo "== $x (time-matched to $g's training loop)"
  wall="$(uv run python -c "import json; print(json.load(open('$RUNS/$g/run_manifest.json'))['timing']['train_wall_seconds'])")"
  echo "matching $g train_wall_seconds = $wall"
  uv run microreason train --config "configs/extra_sft_v3_seed$s.yaml" --parent "$PARENT" \
    --device "$DEVICE" --run-dir "$RUNS/$x" --max-wall-seconds "$wall" "${paid[@]}"
  xc="$(final_checkpoint "$RUNS/$x")"; echo "$x: $xc"
  run_eval "$x" "$xc" val "$SW"
  run_eval "$x" "$xc" val_heldout_shape "$SW"
  compare v2-sft "$x" val_heldout_shape "$SW"
  compare "$x" "$g" val_heldout_shape "$SW"
  ckpts+=("$g:$gc" "$x:$xc")
  printf '%s\n' "${ckpts[@]#*:}" > "$REPORTS/final_checkpoints.txt"
done

echo "== confirmatory test evaluation (reports/confirmatory_plan_v2.md)"
date -u +%Y-%m-%dT%H:%M:%SZ > "$REPORTS/test_unsealed_at.txt"
for pair in "${ckpts[@]}"; do
  name="${pair%%:*}"; ck="${pair#*:}"
  run_eval "$name" "$ck" test_heldout_shape "$SW" --samples "$SAMPLES"
  run_eval "$name" "$ck" test_id "$SW"
  run_eval "$name" "$ck" test_extrapolation "$SW"
done
echo "== primary H4 comparisons"
for s in 1 2; do
  compare v2-sft "v4-grpo-s$s" test_heldout_shape "$SW"
  compare "v2-extra-sft-s$s" "v4-grpo-s$s" test_heldout_shape "$SW"
done

echo "== H1 / H2 on test_id (v0-base, v1-math uploaded during the session)"
waited=0
until [[ -f artifacts/P3_BASE_UPLOAD_COMPLETE ]] || (( waited >= BASE_WAIT_MINUTES * 60 )); do
  sleep 30; waited=$(( waited + 30 ))
done
if [[ -f artifacts/P3_BASE_UPLOAD_COMPLETE ]]; then
  for ck in "$V0" "$V1"; do
    uv run python -c "from microreason.checkpoint import verify_checkpoint as v; v('$ck')"
  done
  run_eval v0-base "$V0" test_id plain-v1
  run_eval v1-math "$V1" test_id plain-v1
  run_eval v1-math "$V1" test_id "$SW"
  compare v0-base v1-math test_id plain-v1
  compare v1-math v2-sft test_id "$SW"
else
  echo "v0-base / v1-math not uploaded within $BASE_WAIT_MINUTES min; H1/H2 undecided" \
    | tee "$REPORTS/h1_h2_skipped.txt"
fi

echo "== decision rules"
uv run python scripts/v3_h4_decision.py --reports "$REPORTS" --out "$REPORTS/h4_decision.json"

echo "== secondary test comparisons"
for s in 1 2; do
  for suite in test_id test_extrapolation; do
    compare v2-sft "v4-grpo-s$s" "$suite" "$SW"
    compare "v2-extra-sft-s$s" "v4-grpo-s$s" "$suite" "$SW"
    compare v2-sft "v2-extra-sft-s$s" "$suite" "$SW"
  done
  compare v2-sft "v2-extra-sft-s$s" test_heldout_shape "$SW"
done
echo "V3_PHASE_C COMPLETE"
