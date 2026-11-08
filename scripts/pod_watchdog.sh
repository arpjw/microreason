#!/usr/bin/env bash
# Run ON THE POD, first, before anything else. Deletes (terminates) this pod at a hard deadline,
# independent of the local machine or this session staying connected.
#
#   scripts/pod_watchdog.sh check               # verify the pod can manage itself; exit 1 if not
#   scripts/pod_watchdog.sh arm <deadline_epoch> # background timer that terminates the pod
#
# The pod's built-in RUNPOD_API_KEY was refused (403) for self-queries in the first pilot attempt,
# so the watchdog prefers MICROREASON_WATCHDOG_KEY: a key the owner stores as a RunPod Secret and
# injects at pod creation with {{ RUNPOD_SECRET_<name> }}. Its value is never printed.
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/pod_env.sh
source "$here/pod_env.sh"

: "${RUNPOD_POD_ID:?RUNPOD_POD_ID is not set; is this running on a RunPod pod?}"
log=/root/watchdog.log
RPCTL_VERSION=2.15.0

ensure_runpodctl() {
  if [[ -x /root/rpctl/runpodctl ]] && /root/rpctl/runpodctl version | grep -q "$RPCTL_VERSION"; then
    return
  fi
  local base="https://github.com/runpod/runpodctl/releases/download/v$RPCTL_VERSION"
  mkdir -p /root/rpctl
  (
    cd /root/rpctl
    curl -sSLO "$base/runpodctl-linux-amd64.tar.gz"
    curl -sSLO "$base/checksums_${RPCTL_VERSION}_sha256.txt"
    grep " runpodctl-linux-amd64.tar.gz$" "checksums_${RPCTL_VERSION}_sha256.txt" | sha256sum -c -
    tar -xzf runpodctl-linux-amd64.tar.gz runpodctl
  )
}

rpctl() {
  RUNPOD_API_KEY="${MICROREASON_WATCHDOG_KEY:-${RUNPOD_API_KEY:-}}" /root/rpctl/runpodctl "$@"
}

case "${1:-}" in
  check)
    ensure_runpodctl
    if [[ -z "${MICROREASON_WATCHDOG_KEY:-}" ]]; then
      echo "warning: MICROREASON_WATCHDOG_KEY not set; trying the pod's built-in key" >&2
    fi
    out="$(rpctl pod get "$RUNPOD_POD_ID" 2>&1 || true)"
    if ! grep -q "\"$RUNPOD_POD_ID\"" <<<"$out"; then
      echo "error: this pod cannot query itself (${out:0:120}); self-termination unavailable" >&2
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
    key="${MICROREASON_WATCHDOG_KEY:-${RUNPOD_API_KEY:-}}"
    nohup env RUNPOD_API_KEY="$key" bash -c "
      sleep $((deadline - now))
      echo \"\$(date -u +%FT%TZ) deadline reached; terminating pod $RUNPOD_POD_ID\" >> $log
      /root/rpctl/runpodctl pod delete $RUNPOD_POD_ID >> $log 2>&1
    " >/dev/null 2>&1 &
    echo $! > /root/watchdog.pid
    echo "armed: pod $RUNPOD_POD_ID terminates at $(date -u -d "@$deadline" +%FT%TZ) (pid $(cat /root/watchdog.pid))"
    ;;
  grace)
    # After a session ends: terminate in <seconds> unless results were collected
    # (/root/collected is created by the local machine once pod_pull.sh succeeds).
    secs="${2:?usage: pod_watchdog.sh grace <seconds>}"
    key="${MICROREASON_WATCHDOG_KEY:-${RUNPOD_API_KEY:-}}"
    nohup env RUNPOD_API_KEY="$key" bash -c "
      sleep $secs
      if [ ! -f /root/collected ]; then
        echo \"\$(date -u +%FT%TZ) grace period over, results not collected; terminating\" >> $log
        /root/rpctl/runpodctl pod delete $RUNPOD_POD_ID >> $log 2>&1
      fi
    " >/dev/null 2>&1 &
    echo "grace armed: pod terminates in ${secs}s unless /root/collected exists"
    ;;
  *)
    echo "usage: pod_watchdog.sh check | arm <deadline_epoch> | grace <seconds>" >&2
    exit 2
    ;;
esac
