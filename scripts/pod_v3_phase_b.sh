#!/usr/bin/env bash
# Run ON THE POD after `SETUP_PROFILE=v3 scripts/pod_setup.sh` (data v3 built and frozen,
# tokenizer v2 trained, datasets tokenized). Protocol v3 Phase B (reports/protocol_v3.md):
#   v0-base -> v1-math -> v2-sft, val and held-out-val evaluations, then the capability gate.
# Evaluation: data v3 suites, cached generator (eval-v2), 320 new tokens. Test suites stay sealed.
set -euo pipefail
cd "${MICROREASON_ROOT:-/root/microreason}"
# shellcheck source=scripts/pod_env.sh
source scripts/pod_env.sh

BUDGET="${BUDGET:-configs/budget_v3_phase_b.yaml}"
DEVICE="${DEVICE:-cuda}"
RUNS="${RUNS:-runs/p3}"
REPORTS="${REPORTS:-reports/v3_phase_b}"
EVAL_BATCH="${EVAL_BATCH:-128}"
SAMPLES="${SAMPLES:-4}"
MAX_NEW="${MAX_NEW:-320}"

: "${MICROREASON_BILLING_START:?set MICROREASON_BILLING_START}"
: "${RUNPOD_POD_ID:?RUNPOD_POD_ID is not set}"
if ! { [[ -f /root/watchdog.pid ]] && kill -0 "$(cat /root/watchdog.pid)"; }; then
  echo "error: watchdog is not armed" >&2
  exit 1
fi
paid=(--budget "$BUDGET" --pod-id "$RUNPOD_POD_ID" --billing-start "$MICROREASON_BILLING_START")
mkdir -p "$RUNS/eval" "$REPORTS"
TOK=artifacts/tokenizer/v2

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
  if [[ "$suite" == test* ]]; then
    echo "error: test suites stay sealed in Phase B" >&2
    exit 1
  fi
  uv run microreason eval --suite "$suite" --checkpoint "$ckpt" --format "$fmt" \
    --device "$DEVICE" --batch-size "$EVAL_BATCH" --data-version v3 --kv-cache \
    --max-new-tokens "$MAX_NEW" --tokenizer "$TOK" --out "$RUNS/eval/${name}_${suite}_${fmt}" \
    "$@" > "$REPORTS/${name}_${suite}_${fmt}_summary.json"
  echo "$name $suite $fmt evaluated"
}

compare() {  # tag a b suite format
  uv run microreason compare --a "$RUNS/eval/${2}_${4}_${5}" --b "$RUNS/eval/${3}_${4}_${5}" \
    > "$REPORTS/${1}_${4}_${5}_compare.json"
  echo "$1 $4 $5 compared"
}

echo "== quick CUDA checks"
timeout 900 uv run pytest -p no:cacheprovider -rN tests/test_model.py tests/test_training.py \
  tests/test_sft.py tests/test_eval.py tests/test_data_v3.py 2>&1 | tail -1 \
  | tee "$REPORTS/cuda_checks.txt"

echo "== frozen data v3 and tokenizer v2"
uv run python - <<'PY' | tee "$REPORTS/data_v3_frozen.json"
import json
f = json.load(open("data/manifests/math_eval_frozen_v3.json"))
s = json.load(open("data/manifests/math_train_stream_v3.json"))
t = json.load(open("artifacts/tokenizer/v2/tokenizer_manifest.json"))
print(json.dumps({"suites": f["suites"], "stream": {k: s[k] for k in ("sha256", "record_count",
      "shards", "by_difficulty", "distinct_shapes")}, "tokenizer_sha256": t["tokenizer_sha256"],
      "pre_tokenization": t["pre_tokenization"]}, indent=2))
PY

echo "== v0-base pretraining (protocol v3)"
uv run microreason train --config configs/pretrain_v3.yaml --device "$DEVICE" \
  --run-dir "$RUNS/v0-base" "${paid[@]}"
v0="$(final_checkpoint "$RUNS/v0-base")"; echo "v0-base: $v0"

echo "== v1-math mid-training (protocol v3)"
uv run microreason train --config configs/midtrain_v3.yaml --device "$DEVICE" --parent "$v0" \
  --run-dir "$RUNS/v1-math" "${paid[@]}"
v1="$(final_checkpoint "$RUNS/v1-math")"; echo "v1-math: $v1"

echo "== plain-v1 val evaluation + H1"
run_eval v0-base "$v0" val plain-v1
run_eval v1-math "$v1" val plain-v1
compare h1 v0-base v1-math val plain-v1

echo "== v2-sft supervised fine-tuning (protocol v3)"
uv run microreason train --config configs/sft_v3.yaml --device "$DEVICE" --parent "$v1" \
  --run-dir "$RUNS/v2-sft" "${paid[@]}"
v2="$(final_checkpoint "$RUNS/v2-sft")"; echo "v2-sft: $v2"
printf '%s\n%s\n%s\n' "$v0" "$v1" "$v2" > "$REPORTS/final_checkpoints.txt"

echo "== SFT-format val evaluation + H2"
run_eval v2-sft "$v2" val sft-show-work-v1 --samples "$SAMPLES"
run_eval v2-sft "$v2" val sft-direct-v1
run_eval v1-math "$v1" val sft-show-work-v1
compare h2 v1-math v2-sft val sft-show-work-v1

echo "== held-out-val (development held-out shapes)"
run_eval v2-sft "$v2" val_heldout_shape sft-show-work-v1 --samples "$SAMPLES"
run_eval v2-sft "$v2" val_heldout_shape sft-direct-v1
run_eval v1-math "$v1" val_heldout_shape plain-v1

echo "== capability gate (pre-declared)"
set +e
uv run python scripts/v3_capability_gate.py \
  --summary "$REPORTS/v2-sft_val_heldout_shape_sft-show-work-v1_summary.json" \
  --out "$REPORTS/capability_gate.json"
gate=$?
set -e
if [[ $gate -eq 0 ]]; then
  echo "CAPABILITY GATE PASSED"
else
  echo "CAPABILITY GATE NOT PASSED (exit $gate): Phase C does not run, per the declared rule"
fi
echo "V3_PHASE_B COMPLETE"
