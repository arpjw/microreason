#!/usr/bin/env bash
# Run ON THE POD after `SETUP_PROFILE=v4x scripts/pod_setup.sh`, after the frozen v4x suite hash
# has been committed (data/manifests/math_eval_frozen_v4x.json at HEAD) and the protocol v4
# checkpoints uploaded to artifacts/v4x/<name>. Protocol v4 replication
# (reports/replication_v4x.md): evaluation only, no training.
set -euo pipefail
cd "${MICROREASON_ROOT:-/root/microreason}"
# shellcheck source=scripts/pod_env.sh
source scripts/pod_env.sh

DEVICE="${DEVICE:-cuda}"
RUNS="${RUNS:-runs/v4x}"
REPORTS="${REPORTS:-reports/replication_v4x}"
EVAL_BATCH="${EVAL_BATCH:-256}"
TOK=artifacts/tokenizer/v2
SW=sft-show-work-v1
NAMES=(v2-sft-long v4-grpo-s1 v4-grpo-s2 v4-grpo-s3 v2-extra-sft-s1 v2-extra-sft-s2
       v2-extra-sft-s3 v4-rft-s1 v4-grpo-step-s1)

if ! { [[ -f /root/watchdog.pid ]] && kill -0 "$(cat /root/watchdog.pid)"; }; then
  echo "error: watchdog is not armed" >&2
  exit 1
fi
mkdir -p "$RUNS" "$REPORTS"
# the frozen file is the committed one (shipped by pod_push); the suite file was rebuilt on the pod
uv run python -c "import hashlib,json; f=json.load(open('data/manifests/math_eval_frozen_v4x.json'))['suites']['test_extrapolation']; \
m=json.load(open('data/manifests/math_test_extrapolation_v4x.json')); h=hashlib.sha256(open(m['file'],'rb').read()).hexdigest(); \
assert h == f['sha256'], 'v4x suite differs from the committed freeze'; print('v4x suite matches the committed freeze', h[:12])"
for n in "${NAMES[@]}"; do
  uv run python -c "from microreason.checkpoint import verify_checkpoint as v; v('artifacts/v4x/$n')"
done
date -u +%Y-%m-%dT%H:%M:%SZ > "$REPORTS/unsealed_at.txt"
for n in "${NAMES[@]}"; do
  uv run microreason eval --suite test_extrapolation --data-version v4x --allow-test \
    --checkpoint "artifacts/v4x/$n" --format "$SW" --device "$DEVICE" --batch-size "$EVAL_BATCH" \
    --kv-cache --max-new-tokens 320 --tokenizer "$TOK" \
    --out "$RUNS/${n}_test_extrapolation_v4x_${SW}" \
    > "$REPORTS/${n}_test_extrapolation_v4x_${SW}_summary.json"
  echo "$n evaluated"
done
uv run python scripts/v4_decide.py replication --eval-dir "$RUNS" --out "$REPORTS/decision.json"
echo "V4X_REPLICATION COMPLETE"
