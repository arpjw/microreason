#!/usr/bin/env bash
# Run ON THE POD after `SETUP_PROFILE=v4c scripts/pod_setup.sh` (v3 suites and RL pool v4
# regenerated against their committed hashes) and after these uploads:
#   artifacts/p4-v2-sft-long (required; the Phase B parent),
#   artifacts/p3-v0-base, artifacts/p3-v1-math, artifacts/p3-v2-sft (H1/H2), then
#   touch artifacts/P4_UPLOAD_COMPLETE.
# Protocol v4 Phase C (reports/protocol_v4.md), then the single confirmatory test evaluation
# (reports/confirmatory_plan_v3.md):
#   parent val re-eval -> for seeds 1..3: v4-grpo (pilot-selected LR) + time-matched
#   v2-extra-sft (pilot-selected LR) -> secondary arms v4-rft-s1, v4-grpo-step-s1 -> all val
#   evals -> test suites unsealed -> test evals -> scripts/v4_decide.py h4.
set -euo pipefail
cd "${MICROREASON_ROOT:-/root/microreason}"
# shellcheck source=scripts/pod_env.sh
source scripts/pod_env.sh

BUDGET="${BUDGET:-configs/budget_v4_phase_c.yaml}"
DEVICE="${DEVICE:-cuda}"
RUNS="${RUNS:-runs/p4c}"
REPORTS="${REPORTS:-reports/v4_phase_c}"
PARENT="${PARENT:-artifacts/p4-v2-sft-long}"
SELECTION="${SELECTION:-reports/v4_phase_b/selection.json}"
EVAL_BATCH="${EVAL_BATCH:-256}"
MAX_NEW="${MAX_NEW:-320}"
UPLOAD_WAIT_MINUTES="${UPLOAD_WAIT_MINUTES:-30}"
TOK=artifacts/tokenizer/v2
SW=sft-show-work-v1
# Amendment 2 (2026-11-09, during Phase C seed 1, before any Phase C result): GRPO steps run
# ~17-21 s, so the session may approach its 8 h ceiling. A budget stop inside a secondary arm must
# never block the confirmatory test phase. FROM=secondary resumes after the primary runs (from
# final_checkpoints.txt); each secondary arm starts only if the time left covers it plus the test
# phase, and a secondary arm that does not complete is reported and skipped.
FROM="${FROM:-start}"
CEILING_HOURS="${CEILING_HOURS:-8}"
RESERVE_SECONDS="${RESERVE_SECONDS:-1200}"   # the budget's reserve_minutes
TEST_PHASE_SECONDS="${TEST_PHASE_SECONDS:-3000}"

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

if [[ "$FROM" == start ]]; then
echo "== quick CUDA checks"
timeout 900 uv run pytest -p no:cacheprovider -rN tests/test_model.py tests/test_rl.py \
  tests/test_rl_v4.py tests/test_eval.py tests/test_v4_decide.py 2>&1 | tail -1 \
  | tee "$REPORTS/cuda_checks.txt"

echo "== parent, pool, selection"
uv run python -c "from microreason.checkpoint import verify_checkpoint as v; \
print('p4-v2-sft-long step', v('$PARENT')['step'])"
uv run python -c "import json; m=json.load(open('data/manifests/math_train_rl_v4.json')); \
print('rl pool v4', m['record_count'], m['sha256'][:12], m['partitions'])"
GRPO_LR="$(uv run python -c "import json; print(json.load(open('$SELECTION'))['grpo']['selected']['lr_text'])")"
SFT_LR="$(uv run python -c "import json; print(json.load(open('$SELECTION'))['extra_sft']['selected']['lr_text'])")"
echo "GRPO lr $GRPO_LR, control lr $SFT_LR (Phase B pilot selection)"

echo "== parent val re-evaluation (same pod and code as the branches)"
run_eval v2-sft-long "$PARENT" val "$SW"
run_eval v2-sft-long "$PARENT" val_heldout_shape "$SW"

ckpts=("v2-sft-long:$PARENT")
for s in 1 2 3; do
  g="v4-grpo-s$s"; x="v2-extra-sft-s$s"
  echo "== $g"
  uv run microreason rl grpo --config "configs/grpo_v4_seed$s.yaml" --parent "$PARENT" \
    --lr "$GRPO_LR" --device "$DEVICE" --run-dir "$RUNS/$g" "${paid[@]}" \
    | tee "$REPORTS/${g}_result.json"
  gc="$(final_checkpoint "$RUNS/$g")"; echo "$g: $gc"
  run_eval "$g" "$gc" val "$SW"
  run_eval "$g" "$gc" val_heldout_shape "$SW"
  wall="$(train_wall "$RUNS/$g")"
  echo "== $x (time-matched to $g's training loop: $wall s)"
  uv run microreason train --config "configs/extra_sft_v4_seed$s.yaml" --parent "$PARENT" \
    --peak-lr "$SFT_LR" --max-wall-seconds "$wall" --device "$DEVICE" --run-dir "$RUNS/$x" \
    "${paid[@]}"
  xc="$(final_checkpoint "$RUNS/$x")"; echo "$x: $xc"
  run_eval "$x" "$xc" val "$SW"
  run_eval "$x" "$xc" val_heldout_shape "$SW"
  ckpts+=("$g:$gc" "$x:$xc")
  printf '%s\n' "${ckpts[@]#*:}" > "$REPORTS/final_checkpoints.txt"
done

else
  # rebuild name:path pairs from the primary runs' final checkpoints
  ckpts=()
  while read -r path; do
    case "$path" in
      artifacts/p4-v2-sft-long) ckpts+=("v2-sft-long:$path") ;;
      *) n="${path#runs/p4c/}"; ckpts+=("${n%%/*}:$path") ;;
    esac
  done < "$REPORTS/final_checkpoints.txt"
  [[ ${#ckpts[@]} -eq 7 ]] || { echo "error: expected 7 primary checkpoints" >&2; exit 1; }
  GRPO_LR="$(uv run python -c "import json; print(json.load(open('$SELECTION'))['grpo']['selected']['lr_text'])")"
  echo "resuming after the primary runs: ${ckpts[*]}"
fi

seconds_left() {
  uv run python -c "import datetime as d, sys; s=d.datetime.fromisoformat(sys.argv[1].replace('Z','+00:00')); \
print(int(s.timestamp() + float(sys.argv[2]) * 3600 - float(sys.argv[3]) - d.datetime.now(d.timezone.utc).timestamp()))" \
    "$MICROREASON_BILLING_START" "$CEILING_HOURS" "$RESERVE_SECONDS"
}

echo "== secondary arms (seed 1): v4-rft-s1, v4-grpo-step-s1"
need="$(( $(uv run python -c "import json; print(int(json.load(open('$RUNS/v4-grpo-s1/run_manifest.json'))['timing']['train_wall_seconds'] * 1.2) + 300)") ))"
for arm in rft:configs/rft_v4_seed1.yaml grpo-step:configs/grpo_step_v4_seed1.yaml; do
  n="v4-${arm%%:*}-s1"
  left="$(seconds_left)"
  if (( left < need + TEST_PHASE_SECONDS )); then
    echo "$n skipped: ${left}s left < ${need}s run + ${TEST_PHASE_SECONDS}s test phase" \
      | tee "$REPORTS/${n}_skipped.txt"
    continue
  fi
  set +e
  uv run microreason rl grpo --config "${arm#*:}" --parent "$PARENT" --lr "$GRPO_LR" \
    --device "$DEVICE" --run-dir "$RUNS/$n" "${paid[@]}" | tee "$REPORTS/${n}_result.json"
  nc="$(final_checkpoint "$RUNS/$n")"
  ok=$?
  set -e
  if [[ $ok -ne 0 ]]; then
    echo "$n did not complete; reported and skipped" | tee "$REPORTS/${n}_skipped.txt"
    continue
  fi
  run_eval "$n" "$nc" val "$SW"
  run_eval "$n" "$nc" val_heldout_shape "$SW"
  ckpts+=("$n:$nc")
  printf '%s\n' "${ckpts[@]#*:}" > "$REPORTS/final_checkpoints.txt"
done

echo "== confirmatory test evaluation (reports/confirmatory_plan_v3.md)"
date -u +%Y-%m-%dT%H:%M:%SZ > "$REPORTS/test_unsealed_at.txt"
for pair in "${ckpts[@]}"; do
  name="${pair%%:*}"; ck="${pair#*:}"
  run_eval "$name" "$ck" test_heldout_shape "$SW" --samples 4
  run_eval "$name" "$ck" test_id "$SW"
  run_eval "$name" "$ck" test_extrapolation "$SW"
done

echo "== H1 / H2 on test_id (protocol v3 Phase B checkpoints, carried over)"
waited=0
until [[ -f artifacts/P4_UPLOAD_COMPLETE ]] || (( waited >= UPLOAD_WAIT_MINUTES * 60 )); do
  sleep 30; waited=$(( waited + 30 ))
done
if [[ -f artifacts/P4_UPLOAD_COMPLETE ]]; then
  for ck in artifacts/p3-v0-base artifacts/p3-v1-math artifacts/p3-v2-sft; do
    uv run python -c "from microreason.checkpoint import verify_checkpoint as v; v('$ck')"
  done
  run_eval v0-base artifacts/p3-v0-base test_id plain-v1
  run_eval v1-math artifacts/p3-v1-math test_id plain-v1
  run_eval v1-math artifacts/p3-v1-math test_id "$SW"
  run_eval v2-sft artifacts/p3-v2-sft test_id "$SW"
else
  echo "protocol v3 checkpoints not uploaded within $UPLOAD_WAIT_MINUTES min; H1/H2 undecided" \
    | tee "$REPORTS/h1_h2_skipped.txt"
fi

echo "== decision rules"
uv run python scripts/v4_decide.py h4 --eval-dir "$RUNS/eval" --out "$REPORTS/h4_decision.json"
echo "V4_PHASE_C COMPLETE"
