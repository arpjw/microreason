#!/usr/bin/env bash
# Run ON THE POD after `SETUP_PROFILE=rl scripts/pod_setup.sh` and after the v2-sft checkpoint
# has been copied to artifacts/p2-v2-sft. Protocol rl-v2 (reports/rl_protocol_v2.md), then the
# confirmatory test evaluation (reports/confirmatory_plan_v1.md):
#   KV-cache parity check -> v2-sft val re-eval -> GRPO v2 at LR 1e-5 and 3e-5 (val evals)
#   -> pre-declared selection -> v2-extra-sft-r2 time-matched to the selected run (val eval)
#   -> test suites unsealed -> primary + secondary test evals -> paired comparisons -> H4 rule.
# Training runs are guarded paid runs under configs/budget_rl_v2.yaml.
set -euo pipefail
cd "${MICROREASON_ROOT:-/root/microreason}"
# shellcheck source=scripts/pod_env.sh
source scripts/pod_env.sh

BUDGET="${BUDGET:-configs/budget_rl_v2.yaml}"
DEVICE="${DEVICE:-cuda}"
RUNS="${RUNS:-runs/rl2}"
REPORTS="${REPORTS:-reports/rl_v2}"
PARENT="${PARENT:-artifacts/p2-v2-sft}"
V1="${V1:-artifacts/rl1}"           # v1 checkpoints for the secondary test evals
V1_WAIT_MINUTES="${V1_WAIT_MINUTES:-20}"
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

run_eval() {  # name checkpoint suite format [extra args]
  local name="$1" ckpt="$2" suite="$3" fmt="$4"; shift 4
  local extra=()
  [[ "$suite" == test* ]] && extra=(--allow-test)
  uv run microreason eval --suite "$suite" --checkpoint "$ckpt" --format "$fmt" \
    --device "$DEVICE" --batch-size "$EVAL_BATCH" --out "$RUNS/eval/${name}_${suite}_${fmt}" \
    "${extra[@]}" "$@" > "$REPORTS/${name}_${suite}_${fmt}_summary.json"
  echo "$name $suite $fmt evaluated"
}

val_branch() {  # name checkpoint
  run_eval "$1" "$2" val sft-show-work-v1 --samples "$SAMPLES"
  run_eval "$1" "$2" val sft-direct-v1
}

compare() {  # a b suite format
  uv run microreason compare --a "$RUNS/eval/${1}_${3}_${4}" --b "$RUNS/eval/${2}_${3}_${4}" \
    > "$REPORTS/${2}_vs_${1}_${3}_${4}_compare.json"
  uv run python -c "import json,sys; c=json.load(open(sys.argv[1])); \
print(sys.argv[1], round(c['difference_b_minus_a'],4), [round(x,4) for x in c['bootstrap95']])" \
    "$REPORTS/${2}_vs_${1}_${3}_${4}_compare.json"
}

val_compare() {  # a b
  compare "$1" "$2" val sft-show-work-v1
  compare "$1" "$2" val sft-direct-v1
}

echo "== quick CUDA checks"
timeout 600 uv run pytest -p no:cacheprovider -rN tests/test_model.py tests/test_rl.py \
  tests/test_eval.py 2>&1 | tail -1 | tee "$REPORTS/cuda_checks.txt"

echo "== parent checkpoint"
uv run python -c "from microreason.checkpoint import verify_checkpoint as v; \
print('v2-sft step', v('$PARENT')['step'])"

echo "== KV-cache sampler parity on v2-sft (512 gate-partition prompts)"
set +e
uv run microreason rl kv-check --config configs/grpo_v2_lr1e-5.yaml --checkpoint "$PARENT" \
  --device "$DEVICE" --out "$REPORTS/kv_check.json" > /dev/null
kv=$?
set -e
uv run python -c "import json; r=json.load(open('$REPORTS/kv_check.json')); \
print({k: r[k] for k in ('passed','fp32_greedy_agreement','bf16_greedy_agreement','pass_at_1_gap','speedup')})"
if [[ $kv -ne 0 ]]; then
  echo "KV CHECK FAILED (exit $kv): no GRPO v2 run, per the declared rule"
  echo "RL_V2 STOPPED_AT_KV_CHECK"
  exit 0
fi

echo "== v2-sft val re-evaluation"
val_branch v2-sft "$PARENT"

for lr in 1e-5 3e-5; do
  name="v4-grpo-v2-lr$lr"
  echo "== $name"
  uv run microreason rl grpo --config "configs/grpo_v2_lr$lr.yaml" --parent "$PARENT" \
    --device "$DEVICE" --run-dir "$RUNS/$name" "${paid[@]}" | tee "$REPORTS/${name}_result.json"
  ck="$(final_checkpoint "$RUNS/$name")"; echo "$name: $ck"
  val_branch "$name" "$ck"
  val_compare v2-sft "$name"
done

echo "== selection (pre-declared: higher val greedy show-work; ties -> lower LR)"
sel="$(uv run python scripts/rl_v2_decide.py select \
  --candidate "v4-grpo-v2-lr1e-5=$REPORTS/v4-grpo-v2-lr1e-5_val_sft-show-work-v1_summary.json" \
  --candidate "v4-grpo-v2-lr3e-5=$REPORTS/v4-grpo-v2-lr3e-5_val_sft-show-work-v1_summary.json" \
  --out "$REPORTS/selection.json")"
echo "selected: $sel"
g="$(final_checkpoint "$RUNS/$sel")"

echo "== v2-extra-sft-r2 (time-matched to the selected run's training loop)"
wall="$(uv run python -c "import json; print(json.load(open('$RUNS/$sel/run_manifest.json'))['timing']['train_wall_seconds'])")"
echo "matching $sel train_wall_seconds = $wall"
uv run microreason train --config configs/extra_sft.yaml --parent "$PARENT" --device "$DEVICE" \
  --run-dir "$RUNS/v2-extra-sft-r2" --max-wall-seconds "$wall" "${paid[@]}"
x="$(final_checkpoint "$RUNS/v2-extra-sft-r2")"; echo "v2-extra-sft-r2: $x"
val_branch v2-extra-sft-r2 "$x"
val_compare v2-sft v2-extra-sft-r2
val_compare v2-extra-sft-r2 "$sel"
printf '%s\n%s\n' "$g" "$x" > "$REPORTS/final_checkpoints.txt"

echo "== confirmatory test evaluation (reports/confirmatory_plan_v1.md)"
date -u +%Y-%m-%dT%H:%M:%SZ > "$REPORTS/test_unsealed_at.txt"
test_branch() {  # name checkpoint
  run_eval "$1" "$2" test_heldout_template sft-show-work-v1 --samples "$SAMPLES"
  run_eval "$1" "$2" test_heldout_template sft-direct-v1
  run_eval "$1" "$2" test_id sft-show-work-v1
  run_eval "$1" "$2" test_extrapolation sft-show-work-v1
}
# Primary: the selected GRPO v2 run is named v4-grpo-r2 in every test artifact.
test_branch v2-sft "$PARENT"
test_branch v4-grpo-r2 "$g"
test_branch v2-extra-sft-r2 "$x"
compare v2-sft v4-grpo-r2 test_heldout_template sft-show-work-v1
compare v2-extra-sft-r2 v4-grpo-r2 test_heldout_template sft-show-work-v1
uv run python scripts/rl_v2_decide.py h4 \
  --vs-sft "$REPORTS/v4-grpo-r2_vs_v2-sft_test_heldout_template_sft-show-work-v1_compare.json" \
  --vs-control "$REPORTS/v4-grpo-r2_vs_v2-extra-sft-r2_test_heldout_template_sft-show-work-v1_compare.json" \
  --out "$REPORTS/h4_decision.json"

echo "== secondary test comparisons"
compare v2-sft v2-extra-sft-r2 test_heldout_template sft-show-work-v1
for suite_fmt in "test_heldout_template sft-direct-v1" "test_id sft-show-work-v1" \
                 "test_extrapolation sft-show-work-v1"; do
  # shellcheck disable=SC2086  # word-splitting suite and format is intended
  compare v2-sft v4-grpo-r2 $suite_fmt
  # shellcheck disable=SC2086
  compare v2-extra-sft-r2 v4-grpo-r2 $suite_fmt
  # shellcheck disable=SC2086
  compare v2-sft v2-extra-sft-r2 $suite_fmt
done

echo "== secondary: v1 checkpoints (uploaded during the session)"
waited=0
until [[ -f "$V1/UPLOAD_COMPLETE" ]] || (( waited >= V1_WAIT_MINUTES * 60 )); do
  sleep 30; waited=$(( waited + 30 ))
done
if [[ -f "$V1/UPLOAD_COMPLETE" ]]; then
  for pair in "v4-grpo:$V1/v4-grpo" "v2-extra-sft:$V1/v2-extra-sft" "v3-dpo:$V1/v3-dpo"; do
    name="${pair%%:*}"; ck="${pair#*:}"
    uv run python -c "from microreason.checkpoint import verify_checkpoint as v; v('$ck')"
    run_eval "$name" "$ck" test_heldout_template sft-show-work-v1
    run_eval "$name" "$ck" test_id sft-show-work-v1
    run_eval "$name" "$ck" test_extrapolation sft-show-work-v1
    for suite in test_heldout_template test_id test_extrapolation; do
      compare v2-sft "$name" "$suite" sft-show-work-v1
    done
  done
  compare v2-extra-sft v4-grpo test_heldout_template sft-show-work-v1
else
  echo "v1 checkpoints not uploaded within $V1_WAIT_MINUTES min; secondary v1 evals skipped" \
    | tee "$REPORTS/v1_secondary_skipped.txt"
fi

echo "RL_V2 COMPLETE"
