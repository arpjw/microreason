#!/usr/bin/env bash
# Run on the local machine. Ship the committed code (git archive of HEAD) to a pod over SSH.
# No credentials are copied: the pod gets a plain source tree plus a REVISION file.
#
#   scripts/pod_push.sh <user@host> <port>
set -euo pipefail

target="${1:?usage: pod_push.sh <user@host> <port>}"
port="${2:?usage: pod_push.sh <user@host> <port>}"
dest=/root/microreason  # container disk: local and fast (/workspace may be network storage)

if [[ -n "$(git status --porcelain)" ]]; then
  echo "error: working tree has uncommitted changes; commit first so the pod runs a known revision" >&2
  exit 1
fi
rev="$(git rev-parse HEAD)"
echo "pushing $rev to $target:$dest"

ssh -p "$port" -o StrictHostKeyChecking=accept-new "$target" "mkdir -p $dest"
git archive --format=tar HEAD | ssh -p "$port" "$target" "tar -x -C $dest"
printf '%s\n' "$rev" | ssh -p "$port" "$target" "cat > $dest/REVISION"
echo "done: $dest at $rev"
