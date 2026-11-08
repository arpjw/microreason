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
uv run microreason data tokenize

echo "== cross-machine determinism check"
uv run python scripts/check_fingerprints.py
