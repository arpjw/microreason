#!/usr/bin/env bash
# Run on the local machine before the pod is terminated. Copies reports, run metrics, manifests,
# budget ledgers, evaluation rows/summaries and the final checkpoints listed in
# reports/stage1/final_checkpoints.txt (if present) into runs/pod_results/<pod-id>/.
#
#   scripts/pod_pull.sh <user@host> <port> <pod-id>
set -euo pipefail
target="${1:?usage}"; port="${2:?usage}"; pod="${3:?usage}"
out="runs/pod_results/$pod"
mkdir -p "$out"
# shellcheck disable=SC2016  # expanded on the pod, not locally
ssh -p "$port" "$target" 'cd /root/microreason && {
    ls -d reports/pilot reports/stage1 2>/dev/null
    for d in runs/*/; do
      for f in run_manifest.json metrics.jsonl budget_ledger.json latest.json; do
        [ -f "$d$f" ] && echo "$d$f"; done; done
    ls runs/eval/*/summary.json runs/eval/*/rows.jsonl 2>/dev/null
    [ -f reports/stage1/final_checkpoints.txt ] && sed "s#^/root/microreason/##" reports/stage1/final_checkpoints.txt
  } | tar -cf - -T - ' | tar -x -C "$out"
ssh -p "$port" "$target" 'cat /root/watchdog.log 2>/dev/null' > "$out/watchdog.log" || true
echo "copied to $out:"
find "$out" -type f | sed "s#^$out/##" | sort | head -60
du -sh "$out"
