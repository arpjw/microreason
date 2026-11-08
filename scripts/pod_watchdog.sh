#!/usr/bin/env bash
# Run ON THE POD, first, before anything else. Deletes (terminates) this pod at a hard deadline,
# independent of the local machine or this session staying connected.
#
#   scripts/pod_watchdog.sh check               # verify the pod can delete itself; exit 1 if not
#   scripts/pod_watchdog.sh arm <deadline_epoch> # background timer that terminates the pod
set -euo pipefail

: "${RUNPOD_POD_ID:?RUNPOD_POD_ID is not set; is this running on a RunPod pod?}"
log=/root/watchdog.log

case "${1:-}" in
  check)
    if ! command -v runpodctl >/dev/null; then
      echo "error: runpodctl is not installed on this pod" >&2
      exit 1
    fi
    # A read of our own pod proves the pod-scoped credentials work for this pod.
    if ! runpodctl pod get "$RUNPOD_POD_ID" >/dev/null 2>&1; then
      echo "error: this pod cannot query itself with runpodctl; self-termination unavailable" >&2
      exit 1
    fi
    echo "ok: pod $RUNPOD_POD_ID can manage itself"
    ;;
  arm)
    deadline="${2:?usage: pod_watchdog.sh arm <deadline_epoch>}"
    now="$(date +%s)"
    if (( deadline <= now )); then
      echo "error: deadline is in the past" >&2
      exit 1
    fi
    nohup bash -c "
      sleep $((deadline - now))
      echo \"\$(date -u +%FT%TZ) deadline reached; terminating pod $RUNPOD_POD_ID\" >> $log
      runpodctl pod delete $RUNPOD_POD_ID >> $log 2>&1
    " >/dev/null 2>&1 &
    echo $! > /root/watchdog.pid
    echo "armed: pod $RUNPOD_POD_ID terminates at $(date -u -d "@$deadline" +%FT%TZ) (pid $(cat /root/watchdog.pid))"
    ;;
  *)
    echo "usage: pod_watchdog.sh check | arm <deadline_epoch>" >&2
    exit 2
    ;;
esac
