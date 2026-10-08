# Shared helpers for the submission scripts. Sourced, not executed.

die() { echo "error: $*" >&2; exit 2; }

repo_root() { git rev-parse --show-toplevel; }

# Export KEY=VALUE lines from a gitignored .env without executing it. Variables already
# set in the environment win, so `HF_BUCKET=x ./scripts/submit_hf.sh` still overrides.
load_env() {
  local f=${1:-.env} line key val
  [[ -f "$f" ]] || return 0
  while IFS= read -r line || [[ -n "$line" ]]; do
    line=${line%$'\r'}
    [[ "$line" =~ ^[[:space:]]*(#|$) ]] && continue
    [[ "$line" =~ ^[[:space:]]*(export[[:space:]]+)?([A-Za-z_][A-Za-z0-9_]*)=(.*)$ ]] || continue
    key=${BASH_REMATCH[2]}
    val=${BASH_REMATCH[3]}
    if [[ "$val" =~ ^\"(.*)\"$ || "$val" =~ ^\'(.*)\'$ ]]; then
      val=${BASH_REMATCH[1]}
    else
      val=$(sed -E 's/[[:space:]]+#.*$//; s/[[:space:]]+$//' <<<"$val")  # inline comment
    fi
    [[ -n "${!key+x}" ]] || export "$key=$val"
  done < "$f"
}

# owner/repo from REPO_SLUG or the origin remote: https://[user@]github.com/o/r[.git],
# git@github.com:o/r[.git] or ssh://git@github.com/o/r[.git].
repo_slug() {
  if [[ -n "${REPO_SLUG:-}" ]]; then echo "$REPO_SLUG"; return; fi
  local url
  url=$(git remote get-url origin 2>/dev/null) || { echo "UNSET_OWNER/UNSET_REPO"; return; }
  url=$(sed -E 's#^(https?://([^@/]+@)?github\.com/|git@github\.com:|ssh://git@github\.com/)##; s#\.git$##; s#/$##' <<<"$url")
  [[ "$url" =~ ^[^/:@]+/[^/:@]+$ ]] || die "cannot parse owner/repo from origin; set REPO_SLUG=owner/repo"
  echo "$url"
}

# Refuse a dirty tree: the pinned SHA would not match what you are testing locally.
require_clean() {
  local allow_dirty=$1
  if [[ -n "$(git status --porcelain)" ]]; then
    if [[ "$allow_dirty" == 1 ]]; then
      echo "warning: working tree is dirty; submitting HEAD $(git rev-parse --short HEAD) anyway" >&2
    else
      git status --short >&2
      die "working tree is dirty; commit first or pass --allow-dirty (the job runs HEAD, not your local changes)"
    fi
  fi
}

# Print a command with shell quoting, one argument per token.
print_cmd() { printf '%q ' "$@"; echo; }
