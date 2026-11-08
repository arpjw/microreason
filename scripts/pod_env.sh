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

# The container sees every host CPU (e.g. 120) but its cgroup quota is far smaller (e.g. 12.75).
# Thread pools sized to the host oversubscribe the quota badly: in the first pilot run the test
# suite spawned 189 threads and ran for 31+ minutes instead of ~30 s. Cap them at the quota.
if [[ -r /sys/fs/cgroup/cpu.max ]]; then
  read -r quota period < /sys/fs/cgroup/cpu.max
  if [[ "$quota" != "max" ]]; then
    cpus=$(( quota / period ))
    (( cpus < 1 )) && cpus=1
    export OMP_NUM_THREADS="$cpus" MKL_NUM_THREADS="$cpus" RAYON_NUM_THREADS="$cpus"
  fi
fi
