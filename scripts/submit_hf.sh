#!/usr/bin/env bash
# Submit one detached HF Job running src.train (or src.evaluate) at the current commit.
#
#   ./scripts/submit_hf.sh [--dry-run] [--allow-dirty] [--flavor F] [--timeout T]
#                          [--entry train|evaluate] <hydra overrides...>
#
# Env: HF_BUCKET (bucket for checkpoints/logs, mounted at /ckpt), HF_TOKEN, WANDB_API_KEY.
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/_common.sh
load_env
# Prefer the project's pinned hf CLI over whatever else is on PATH.
for d in .venv/Scripts .venv/bin; do [[ -d "$d" ]] && PATH="$PWD/$d:$PATH"; done

dry_run=0; allow_dirty=0; flavor=a10g-small; timeout=4h; entry=train
overrides=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) dry_run=1 ;;
    --allow-dirty) allow_dirty=1 ;;
    --flavor) flavor=$2; shift ;;
    --timeout) timeout=$2; shift ;;
    --entry) entry=$2; shift ;;
    --) shift; overrides+=("$@"); break ;;
    *) overrides+=("$1") ;;
  esac
  shift
done
[[ "$entry" == train || "$entry" == evaluate ]] || die "--entry must be train or evaluate"

require_clean "$allow_dirty"
sha=$(git rev-parse HEAD)
slug=$(repo_slug)
bucket=${HF_BUCKET:-}
if [[ -z "$bucket" ]]; then
  [[ "$dry_run" == 1 ]] || die "HF_BUCKET is not set (e.g. export HF_BUCKET=<user>/<bucket>)"
  bucket="<HF_BUCKET>"
fi
[[ "$slug" != UNSET_OWNER/* || "$dry_run" == 1 ]] || die "no origin remote; set REPO_SLUG=owner/repo"

case "$timeout" in
  *d) timeout_s=$(( ${timeout%d} * 86400 )) ;;
  *h) timeout_s=$(( ${timeout%h} * 3600 )) ;;
  *m) timeout_s=$(( ${timeout%m} * 60 )) ;;
  *s) timeout_s=${timeout%s} ;;
  *) timeout_s=$timeout ;;
esac
[[ "$timeout_s" =~ ^[0-9]+$ ]] || die "--timeout must be an integer with optional s/m/h/d suffix"

cmd=(hf jobs uv run
  --flavor "$flavor" --timeout "$timeout" -d
  -s HF_TOKEN -s WANDB_API_KEY
  -e RUN_ENV=hf_jobs -e REPO_SLUG="$slug" -e REPO_SHA="$sha" -e ENTRY="$entry" -e MAX_RUNTIME_S="$timeout_s"
  -v "hf://buckets/$bucket/ckpt:/ckpt"
  "https://raw.githubusercontent.com/$slug/$sha/scripts/hf_entry.py"
  -- "${overrides[@]}")

if [[ "$dry_run" == 1 ]]; then
  print_cmd "${cmd[@]}"
  exit 0
fi

out=$("${cmd[@]}")
echo "$out"
job_id=$(grep -oE '[0-9a-f]{24}' <<<"$out" | head -1 || true)
[[ -n "$job_id" ]] || die "could not parse a job id from the output above"
echo "job_id=$job_id"
echo "follow: hf jobs logs -f $job_id"
