#!/usr/bin/env bash
# Run ON THE POD after pod_push.sh. Installs the environment and rebuilds every dataset from
# pinned sources, then checks the rebuilt data matches the hashes recorded locally.
set -euo pipefail
cd /root/microreason

if ! command -v uv >/dev/null; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi

echo "== environment"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
uv sync --frozen
uv run python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'available', torch.cuda.is_available())"

echo "== data (rebuilt from pinned sources)"
uv run microreason data fetch-sources
uv run microreason data split-text --out /tmp/general_text_split_v1.json
if [[ "${SETUP_PROFILE:-full}" == "v3c" ]]; then
  # Protocol v3 Phase C: regenerate the frozen v3 suites (the build refuses any change), build
  # the RL pool, verify the committed tokenizer v2. No pretraining stream or tokenization needed.
  mkdir -p reports/v3_phase_c
  echo "== data v3 suites (regenerated; must match math_eval_frozen_v3.json)"
  uv run microreason data generate-v3 --config configs/data_v3.yaml > reports/v3_phase_c/data_v3_build.json
  uv run microreason data audit-v3 --report reports/v3_phase_c/data_v3_audit.json > /dev/null
  echo "== RL pool v3"
  uv run microreason data generate-rl-pool-v3 --config configs/data_v3.yaml \
    --pool configs/rl_pool_v3.yaml | tee reports/v3_phase_c/rl_pool_v3.json
  echo "== tokenizer v2 (committed)"
  uv run python -c "import hashlib, json; m = json.load(open('artifacts/tokenizer/v2/tokenizer_manifest.json')); \
h = hashlib.sha256(open('artifacts/tokenizer/v2/tokenizer.json', 'rb').read()).hexdigest(); \
assert h == m['tokenizer_sha256'], 'tokenizer v2 hash mismatch'; print('tokenizer v2 verified', h[:12])"
  echo "== setup v3c complete"
  exit 0
fi
if [[ "${SETUP_PROFILE:-full}" == "v4" || "${SETUP_PROFILE:-full}" == "v4c" ]]; then
  # Protocol v4 (reports/protocol_v4.md): regenerate the frozen v3 suites (the build refuses any
  # change), verify the committed tokenizer v2. Phase B (v4) also builds the v4 SFT pool here;
  # its session script builds the RL pool v4 in the background. Phase C (v4c) rebuilds the RL
  # pool, which must match the manifest committed after Phase B.
  rep="reports/v4_phase_b"; [[ "$SETUP_PROFILE" == "v4c" ]] && rep="reports/v4_phase_c"
  mkdir -p "$rep"
  echo "== data v3 suites (regenerated; must match math_eval_frozen_v3.json)"
  uv run microreason data generate-v3 --config configs/data_v3.yaml > "$rep/data_v3_build.json"
  uv run microreason data audit-v3 --report "$rep/data_v3_audit.json" > /dev/null
  echo "== tokenizer v2 (committed)"
  uv run python -c "import hashlib, json; m = json.load(open('artifacts/tokenizer/v2/tokenizer_manifest.json')); \
h = hashlib.sha256(open('artifacts/tokenizer/v2/tokenizer.json', 'rb').read()).hexdigest(); \
assert h == m['tokenizer_sha256'], 'tokenizer v2 hash mismatch'; print('tokenizer v2 verified', h[:12])"
  if [[ "$SETUP_PROFILE" == "v4" ]]; then
    echo "== SFT pool v4 (5 M records; v2-sft-long uses 3 M of its sft_extra partition)"
    uv run microreason data generate-rl-pool-v3 --config configs/data_v3.yaml \
      --pool configs/sft_pool_v4.yaml | tee "$rep/sft_pool_v4.json"
  else
    committed="$(git show HEAD:data/manifests/math_train_rl_v4.json | uv run python -c "import json,sys; print(json.load(sys.stdin)['sha256'])")"
    echo "== RL pool v4 (must match the Phase B manifest, sha256 ${committed:0:12})"
    uv run microreason data generate-rl-pool-v3 --config configs/data_v3.yaml \
      --pool configs/rl_pool_v4.yaml | tee "$rep/rl_pool_v4.json"
    uv run python -c "import json,sys; m=json.load(open('data/manifests/math_train_rl_v4.json')); \
assert m['sha256'] == sys.argv[1], 'RL pool v4 differs from Phase B'; print('rl pool v4 verified')" "$committed"
  fi
  echo "== setup $SETUP_PROFILE complete"
  exit 0
fi
if [[ "${SETUP_PROFILE:-full}" == "v3" ]]; then
  # Protocol v3 (reports/protocol_v3.md): data v3 + tokenizer v2; no v1 math data is needed.
  echo "== data v3: suites (built, audited, frozen), SFT split, stream"
  mkdir -p reports/v3_phase_b
  uv run microreason data generate-v3 --config configs/data_v3.yaml > reports/v3_phase_b/data_v3_build.json
  uv run microreason data audit-v3 --report reports/v3_phase_b/data_v3_audit.json > /dev/null
  uv run microreason data generate-stream-v3 --config configs/data_v3.yaml
  echo "== tokenizer v2 (one token per digit)"
  if [[ -f artifacts/tokenizer/v2/tokenizer.json ]]; then
    # Committed after Phase B: reuse it, never retrain (checkpoints are bound to its hash).
    uv run python -c "import hashlib, json; m = json.load(open('artifacts/tokenizer/v2/tokenizer_manifest.json')); \
h = hashlib.sha256(open('artifacts/tokenizer/v2/tokenizer.json', 'rb').read()).hexdigest(); \
assert h == m['tokenizer_sha256'], 'tokenizer v2 hash mismatch'; print('tokenizer v2 verified', h[:12])"
  else
    uv run microreason tokenizer train --config configs/tokenizer_v2.yaml
  fi
  echo "== tokenize with tokenizer v2"
  uv run microreason data tokenize --tokenizer artifacts/tokenizer/v2 --out data/tokenized_v2 \
    --datasets general_text_train_v1,general_text_val_v1,math_train_v3,math_val_v3
  uv run microreason data tokenize-stream-v3 --config configs/data_v3.yaml
  echo "== setup v3 complete"
  exit 0
fi
uv run microreason data generate          # refuses to change the frozen evaluation suites
uv run microreason data audit --report /tmp/data_audit_v1.json
uv run microreason data freeze             # idempotent: verifies the frozen suite hashes
if [[ "${SETUP_PROFILE:-full}" == "scale" ]]; then
  # Protocol v2: base datasets + stream v2 (stream v1 is not needed).
  uv run microreason data tokenize \
    --datasets general_text_train_v1,general_text_val_v1,math_train_v1,math_val_v1
  echo "== cross-machine determinism check (base data)"
  uv run python scripts/check_fingerprints.py --skip-stream
  echo "== stream v2"
  uv run microreason data generate-stream --config configs/data_stream_v2.yaml
  uv run microreason data audit-stream --config configs/data_stream_v2.yaml --sample 0.005 \
    --report /tmp/data_stream_audit_v2.json
  uv run microreason data tokenize-stream --config configs/data_stream_v2.yaml
  exit 0
fi
if [[ "${SETUP_PROFILE:-full}" == "rl" ]]; then
  # RL branches: the math splits plus the training-only rl1 pool (no tokenized streams needed).
  echo "== cross-machine determinism check (math manifests)"
  uv run python scripts/check_fingerprints.py --only math
  echo "== stream rl1 (RL pool)"
  uv run microreason data generate-stream --config configs/data_stream_rl.yaml
  uv run microreason data audit-stream --config configs/data_stream_rl.yaml --sample 0.02 \
    --report /tmp/data_stream_audit_rl1.json
  exit 0
fi
if [[ "${SETUP_PROFILE:-full}" == "sft" ]]; then
  # SFT and its evaluations need only the math splits (already rebuilt and audited above).
  echo "== cross-machine determinism check (math manifests)"
  uv run python scripts/check_fingerprints.py --only math
  exit 0
fi
uv run microreason data generate-stream    # training-only math stream (excludes frozen suites)
uv run microreason data audit-stream --report /tmp/data_stream_audit_v1.json
uv run microreason data tokenize

echo "== cross-machine determinism check"
uv run python scripts/check_fingerprints.py
