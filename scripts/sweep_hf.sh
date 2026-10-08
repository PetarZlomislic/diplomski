#!/usr/bin/env bash
# Submit one detached HF Job per combination of comma-separated override values.
#
#   ./scripts/sweep_hf.sh [submit_hf.sh flags...] model=a,b seed=0,1 optim.lr=3e-4
#
# Commas inside [...] or {...} are not split. Job ids go to outputs/sweeps/<ts>.tsv.
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/_common.sh

flags=(); axes=(); dry_run=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) flags+=("$1"); dry_run=1 ;;
    --allow-dirty) flags+=("$1") ;;
    --flavor|--timeout|--entry) flags+=("$1" "$2"); shift ;;
    *) axes+=("$1") ;;
  esac
  shift
done
[[ ${#axes[@]} -gt 0 ]] || die "no overrides given"

# key=v1,v2 -> newline-separated key=v1 / key=v2, splitting only at bracket depth 0.
expand() {
  local arg=$1 key=${1%%=*} vals=${1#*=} depth=0 cur="" ch i
  [[ "$arg" == *=* ]] || { echo "$arg"; return; }
  for (( i=0; i<${#vals}; i++ )); do
    ch=${vals:i:1}
    case "$ch" in
      "["|"{") depth=$((depth+1)) ;;
      "]"|"}") depth=$((depth-1)) ;;
    esac
    if [[ "$ch" == "," && $depth -eq 0 ]]; then echo "$key=$cur"; cur=""; else cur+=$ch; fi
  done
  echo "$key=$cur"
}

combos=("")
for axis in "${axes[@]}"; do
  mapfile -t values < <(expand "$axis")
  next=()
  for c in "${combos[@]}"; do
    for v in "${values[@]}"; do next+=("${c:+$c$'\x1f'}$v"); done
  done
  combos=("${next[@]}")
done

manifest=""
if [[ "$dry_run" == 0 ]]; then
  mkdir -p outputs/sweeps
  manifest="outputs/sweeps/$(date -u +%Y%m%d-%H%M%S).tsv"
  printf 'job_id\toverrides\n' > "$manifest"
fi

for c in "${combos[@]}"; do
  IFS=$'\x1f' read -r -a ov <<<"$c"
  if [[ "$dry_run" == 1 ]]; then
    ./scripts/submit_hf.sh "${flags[@]}" -- "${ov[@]}"
  else
    out=$(./scripts/submit_hf.sh "${flags[@]}" -- "${ov[@]}")
    echo "$out"
    job_id=$(sed -n 's/^job_id=//p' <<<"$out")
    printf '%s\t%s\n' "$job_id" "${ov[*]}" >> "$manifest"
  fi
done
[[ -z "$manifest" ]] || echo "manifest: $manifest"
