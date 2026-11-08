#!/usr/bin/env bash
# Sourced ON THE POD by the other pod scripts. SSH sessions do not inherit the container's
# environment, so load the RunPod variables from PID 1. Values are never printed.
if [[ -z "${RUNPOD_POD_ID:-}" && -r /proc/1/environ ]]; then
  while IFS= read -r -d '' kv; do
    case "$kv" in
      RUNPOD_POD_ID=*|RUNPOD_API_KEY=*|MICROREASON_WATCHDOG_KEY=*) export "${kv?}" ;;
    esac
  done < /proc/1/environ
fi
export PATH="/root/rpctl:$HOME/.local/bin:$PATH"
