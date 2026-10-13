#!/usr/bin/env bash
# Prepare reBEN tiers on an HF Job: download the authors' LMDB (~155 GB) to the job's
# disk, write the dev / sweep / full litdata tiers and push each to <prefix>-<tier>.
# Nothing is stored on this machine.
#
#   ./scripts/prepare_hf.sh <user>/reben [--dry-run] [extra prepare_data flags...]
#   e.g. ./scripts/prepare_hf.sh <user>/reben --fraction 0.15     # only the sweep tier
#
# cpu-xl: 16 vCPU, 124 GB RAM, 1000 GB disk, ~$1/h. The full run (download, three tiers,
# ~300 GB of uploads) takes several hours; the timeout leaves room for it.
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/_common.sh

[[ $# -ge 1 && "$1" != -* ]] || die "usage: $0 <hf-user>/<dataset-prefix> [--dry-run] [flags]"
prefix=$1; shift
submit=(); extra=()
for a in "$@"; do
  case "$a" in
    --dry-run|--allow-dirty) submit+=("$a") ;;
    *) extra+=("$a") ;;
  esac
done
exec ./scripts/submit_hf.sh "${submit[@]}" --entry prepare_data --flavor cpu-xl --timeout 24h \
  -- --source reben --out-repo "$prefix" "${extra[@]}"
