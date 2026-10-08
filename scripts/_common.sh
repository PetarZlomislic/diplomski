# Shared helpers for the submission scripts. Sourced, not executed.

die() { echo "error: $*" >&2; exit 2; }

repo_root() { git rev-parse --show-toplevel; }

# owner/repo from REPO_SLUG or the origin remote (https or ssh form).
repo_slug() {
  if [[ -n "${REPO_SLUG:-}" ]]; then echo "$REPO_SLUG"; return; fi
  local url
  url=$(git remote get-url origin 2>/dev/null) || { echo "UNSET_OWNER/UNSET_REPO"; return; }
  url=${url%.git}
  url=${url#git@github.com:}
  url=${url#https://github.com/}
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
