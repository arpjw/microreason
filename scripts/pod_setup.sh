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
