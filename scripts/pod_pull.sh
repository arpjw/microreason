#!/usr/bin/env bash
# Run on the local machine before the pod is terminated. Copies reports, metrics, run manifests
# and budget ledgers (not checkpoints) into runs/pod_results/<pod-id>/.
#
#   scripts/pod_pull.sh <user@host> <port> <pod-id>
set -euo pipefail
target="${1:?usage}"; port="${2:?usage}"; pod="${3:?usage}"
out="runs/pod_results/$pod"
mkdir -p "$out"
ssh -p "$port" "$target" 'cd /root/microreason && tar -cf - reports/pilot \
    $(ls -d runs/*/ 2>/dev/null | while read -r d; do
        for f in run_manifest.json metrics.jsonl budget_ledger.json latest.json; do
          [ -f "$d$f" ] && echo "$d$f"; done; done) \
    /root/watchdog.log 2>/dev/null' | tar -x -C "$out"
echo "copied to $out:"
find "$out" -type f | sort
