#!/usr/bin/env bash
# Submit one detached HF Job running src.train (or src.evaluate) at the current commit.
#
#   ./scripts/submit_hf.sh [--dry-run] [--allow-dirty] [--flavor F] [--timeout T]
#                          [--entry train|evaluate|sweep_eval|prepare_data]
#                          [--data <owner>/<dataset>]... <hydra overrides...>
#
# Hub datasets are DOWNLOADED to the job's disk at /datasets/<owner>/<dataset> (the HF Jobs
# data root) before the entrypoint runs: one given with --data, and the one named by a
# relative `data.data_uri=<owner>/<dataset>`. Evaluation jobs read the data_uri from the run
# config, so pass --data for them. Mind the flavor's disk (a10g-small: 110 GB).
#
# Env: HF_BUCKET (bucket for checkpoints/logs, mounted at /ckpt), HF_TOKEN, WANDB_API_KEY.
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/_common.sh
load_env
# Prefer the project's pinned hf CLI over whatever else is on PATH.
for d in .venv/Scripts .venv/bin; do [[ -d "$d" ]] && PATH="$PWD/$d:$PATH"; done

dry_run=0; allow_dirty=0; flavor=a10g-small; timeout=4h; entry=train
overrides=(); datasets=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) dry_run=1 ;;
    --allow-dirty) allow_dirty=1 ;;
    --flavor) flavor=$2; shift ;;
    --timeout) timeout=$2; shift ;;
    --entry) entry=$2; shift ;;
    --data) datasets+=("$2"); shift ;;
    --) shift; overrides+=("$@"); break ;;
    *) overrides+=("$1") ;;
  esac
  shift
done
[[ "$entry" =~ ^(train|evaluate|sweep_eval|prepare_data)$ ]] || die "--entry must be train, evaluate, sweep_eval or prepare_data"

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

for o in "${overrides[@]}"; do
  if [[ "$o" =~ ^data\.data_uri=([A-Za-z0-9._-]+/[A-Za-z0-9._-]+)$ ]]; then
    datasets+=("${BASH_REMATCH[1]}")
  fi
done
repos=""
for d in $(printf '%s
' "${datasets[@]}" | sort -u); do
  [[ "$d" =~ ^[A-Za-z0-9._-]+/[A-Za-z0-9._-]+$ ]] || die "--data must be <owner>/<dataset>, got $d"
  repos+="${repos:+ }$d"
done
data_env=()
[[ -n "$repos" ]] && data_env=(-e "DATASETS=$repos")

# hf refuses to forward a secret that is unset locally, so W&B's is optional.
secrets=(-s HF_TOKEN)
[[ -n "${WANDB_API_KEY:-}" ]] && secrets+=(-s WANDB_API_KEY)

cmd=(hf jobs uv run
  --flavor "$flavor" --timeout "$timeout" -d --json
  "${secrets[@]}"
  -e RUN_ENV=hf_jobs -e REPO_SLUG="$slug" -e REPO_SHA="$sha" -e ENTRY="$entry" -e MAX_RUNTIME_S="$timeout_s"
  "${data_env[@]}" -v "hf://buckets/$bucket/ckpt:/ckpt"
  "https://raw.githubusercontent.com/$slug/$sha/scripts/hf_entry.py"
  -- "${overrides[@]}")

if [[ "$dry_run" == 1 ]]; then
  print_cmd "${cmd[@]}"
  exit 0
fi

out=$("${cmd[@]}")
echo "$out"
# The --json result carries the job URL .../jobs/<owner>/<id>; take owner/id from it only,
# never from other hex strings in the output (e.g. the pinned commit in the script URL).
job_id=$(grep -oE 'huggingface\.co/jobs/[^/"[:space:]]+/[0-9a-f]{24}\b' <<<"$out" \
  | head -1 | sed 's#.*/jobs/##' || true)
[[ -n "$job_id" ]] || die "could not find the job URL in the output above"
echo "job_id=$job_id"
echo "follow: uv run hf jobs logs -f $job_id"
