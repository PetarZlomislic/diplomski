#!/usr/bin/env bash
# Push a Kaggle kernel that runs src.train at the current commit, wait for it, pull outputs.
#
#   ./scripts/submit_kaggle.sh [--dry-run] [--allow-dirty] [--entry train|evaluate|exp1_run]
#                              [--kernel user/slug] [--data owner/dataset]... <overrides...>
#
# Env: KAGGLE_KERNEL (user/slug) unless --kernel is given; Kaggle CLI credentials.
# Hub datasets (--data, and a relative `data.data_uri=<owner>/<dataset>`) are downloaded to
# the kernel's scratch disk before the entrypoint. A private one needs a Kaggle Secret
# HF_TOKEN attached to the kernel.
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/_common.sh
load_env

dry_run=0; allow_dirty=0; entry=train; kernel=${KAGGLE_KERNEL:-}; poll_s=30
overrides=(); datasets=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) dry_run=1 ;;
    --allow-dirty) allow_dirty=1 ;;
    --entry) entry=$2; shift ;;
    --kernel) kernel=$2; shift ;;
    --data) datasets+=("$2"); shift ;;
    --) shift; overrides+=("$@"); break ;;
    *) overrides+=("$1") ;;
  esac
  shift
done
[[ "$entry" =~ ^(train|evaluate|sweep_eval|exp1_run)$ ]] \
  || die "--entry must be train, evaluate, sweep_eval or exp1_run"
for o in "${overrides[@]}"; do
  if [[ "$o" =~ data\.data_uri=([A-Za-z0-9._-]+/[A-Za-z0-9._-]+)([],]|$) ]]; then
    datasets+=("${BASH_REMATCH[1]}")
  fi
done
repos=""
for d in $(printf '%s\n' "${datasets[@]}" | sort -u); do
  [[ "$d" =~ ^[A-Za-z0-9._-]+/[A-Za-z0-9._-]+$ ]] || die "--data must be <owner>/<dataset>, got $d"
  repos+="${repos:+ }$d"
done

require_clean "$allow_dirty"
sha=$(git rev-parse HEAD)
slug=$(repo_slug)
if [[ -z "$kernel" ]]; then
  [[ "$dry_run" == 1 ]] || die "set KAGGLE_KERNEL=<user>/<slug> or pass --kernel"
  kernel="<KAGGLE_USER>/robustness-eval-$entry"
fi
[[ "$slug" != UNSET_OWNER/* || "$dry_run" == 1 ]] || die "no origin remote; set REPO_SLUG=owner/repo"

# Overrides travel as one shell-quoted string, split again by src/env.py (shlex).
hydra_overrides=$(printf '%q ' "${overrides[@]}")
hydra_overrides=${hydra_overrides% }
safe_kernel=$(printf %s "$kernel" | tr -c 'A-Za-z0-9._-' _)
build=outputs/kaggle_build/$safe_kernel
mkdir -p "$build"
PY=$(command -v python3 || command -v python) || die "python is needed to stamp the kernel"
REPO_SLUG="$slug" REPO_SHA="$sha" HYDRA_OVERRIDES="$hydra_overrides" ENTRY="$entry" DATASETS="$repos" \
KERNEL_ID="$kernel" BUILD="$build" "$PY" - <<'PYEOF'
import json, os, pathlib
b = pathlib.Path(os.environ["BUILD"])
src = pathlib.Path("kernels/train/run.py").read_text()
for k in ("REPO_SLUG", "REPO_SHA", "HYDRA_OVERRIDES", "ENTRY", "DATASETS"):
    src = src.replace(json.dumps(f"__{k}__"), json.dumps(os.environ[k]))
(b / "run.py").write_text(src)
meta = json.loads(pathlib.Path("kernels/train/kernel-metadata.json").read_text())
meta["id"] = os.environ["KERNEL_ID"]
meta["title"] = os.environ["KERNEL_ID"].split("/")[-1]
(b / "kernel-metadata.json").write_text(json.dumps(meta, indent=2))
PYEOF

out_dir="outputs/kaggle/$safe_kernel/$sha"
push_cmd=(kaggle kernels push -p "$build")
if [[ "$dry_run" == 1 ]]; then
  echo "# kernel $kernel at $slug@$sha, RUN_ENV=kaggle HYDRA_OVERRIDES=$hydra_overrides DATASETS=$repos"
  print_cmd "${push_cmd[@]}"
  print_cmd kaggle kernels status "$kernel"
  print_cmd kaggle kernels output "$kernel" -p "$out_dir"
  exit 0
fi

command -v kaggle >/dev/null || die "kaggle CLI not found (uv tool install kaggle)"
export PYTHONUTF8=1  # the Kaggle CLI crashes printing non-ASCII on Windows consoles otherwise
"${push_cmd[@]}"
echo "pushed $kernel; polling every ${poll_s}s"
while true; do
  status=$(kaggle kernels status "$kernel" 2>&1 || true)
  echo "$(date -u +%H:%M:%S) $status"
  case "$status" in
    *COMPLETE*|*complete*) break ;;
    *ERROR*|*error*|*CANCEL*|*cancel*) echo "kernel ended: $status" >&2; break ;;
  esac
  sleep "$poll_s"
done
mkdir -p "$out_dir"
kaggle kernels logs "$kernel" > "$out_dir/kernel_log.json" 2>&1 || true
# `kernels output` can hang when a failed kernel produced no files; bound it.
timeout 600 kaggle kernels output "$kernel" -p "$out_dir" || echo "warning: output download failed or timed out" >&2
echo "outputs: $out_dir (execution log: $out_dir/kernel_log.json)"
[[ "$status" == *COMPLETE* || "$status" == *complete* ]]
