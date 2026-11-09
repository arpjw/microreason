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
ssh -p "$port" "$target" "PULL_ROLLOUTS=${PULL_ROLLOUTS:-0}; "'cd /root/microreason && {
    ls -d reports/pilot reports/stage1 reports/stage2 reports/scale_v2 reports/rl_v1 reports/rl_v2 reports/v3_phase_b reports/v3_phase_c reports/v4_phase_b reports/v4_phase_c artifacts/tokenizer/v2 2>/dev/null
    ls data/manifests/math_*_v3.json data/manifests/math_*_v4.json 2>/dev/null
    for d in runs/*/ runs/p2/*/ runs/rl/*/ runs/rl2/*/ runs/p3/*/ runs/p3c/*/ runs/p4/*/ runs/p4c/*/; do
      for f in run_manifest.json metrics.jsonl budget_ledger.json latest.json rollouts.jsonl pairs.jsonl; do
        # protocol v4 rollout records are ~0.5-0.8 GB per run: their sha256 is in each run
        # manifest; copy them only when asked (PULL_ROLLOUTS=1)
        case "$d$f" in runs/p4*/rollouts.jsonl) [ "${PULL_ROLLOUTS:-0}" = 1 ] || continue;; esac
        [ -f "$d$f" ] && echo "$d$f"; done; done
    ls runs/eval/*/summary.json runs/eval/*/rows.jsonl runs/p2/eval/*/summary.json runs/p2/eval/*/rows.jsonl runs/rl/eval/*/summary.json runs/rl/eval/*/rows.jsonl runs/rl2/eval/*/summary.json runs/rl2/eval/*/rows.jsonl runs/p3/eval/*/summary.json runs/p3/eval/*/rows.jsonl runs/p3c/eval/*/summary.json runs/p3c/eval/*/rows.jsonl runs/p4/eval/*/summary.json runs/p4/eval/*/rows.jsonl runs/p4c/eval/*/summary.json runs/p4c/eval/*/rows.jsonl 2>/dev/null
    cat reports/stage*/final_checkpoints.txt reports/scale_*/final_checkpoints.txt reports/rl_*/final_checkpoints.txt reports/v3_*/final_checkpoints.txt reports/v4_*/final_checkpoints.txt 2>/dev/null | sed "s#^/root/microreason/##"
  } | tar -cf - -T - ' | tar -x -C "$out"
ssh -p "$port" "$target" 'cat /root/watchdog.log 2>/dev/null' > "$out/watchdog.log" || true
echo "copied to $out:"
find "$out" -type f | sed "s#^$out/##" | sort | head -60
du -sh "$out"
