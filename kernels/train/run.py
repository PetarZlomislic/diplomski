"""Kaggle kernel shim: fetch the repo at a pinned SHA, sync the LOCKED env, run an entrypoint.

scripts/submit_kaggle.sh stamps the constants below into a build copy of this file; Kaggle
pushes a single code file and has no per-run environment, so they travel inside it.

The project runs in its own uv venv, not on top of Kaggle's preinstalled packages: those
include extras (e.g. transformers) that conflict with the pinned versions once imported.
"""
import os
import subprocess
import sys
import tarfile
import urllib.request
from pathlib import Path

REPO_SLUG = "__REPO_SLUG__"
REPO_SHA = "__REPO_SHA__"
HYDRA_OVERRIDES = "__HYDRA_OVERRIDES__"
ENTRY = "__ENTRY__"


def fetch(dest: Path) -> Path:
    archive = dest / "repo.tar.gz"
    req = urllib.request.Request(f"https://api.github.com/repos/{REPO_SLUG}/tarball/{REPO_SHA}")
    with urllib.request.urlopen(req, timeout=120) as r, archive.open("wb") as f:
        f.write(r.read())
    with tarfile.open(archive) as tar:
        tar.extractall(dest, filter="data")
    return next(p for p in dest.iterdir() if p.is_dir() and p.name.endswith(REPO_SHA[:7]))


def main() -> int:
    dest = Path("/kaggle/tmp") if Path("/kaggle").exists() else Path.cwd() / "kaggle_tmp"
    dest.mkdir(parents=True, exist_ok=True)
    root = fetch(dest)

    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "uv"], check=True)
    uv = [sys.executable, "-m", "uv"]
    subprocess.run([*uv, "sync", "--frozen", "--extra", "logging"], cwd=root, check=True)

    env = os.environ | {"RUN_ENV": "kaggle", "HYDRA_OVERRIDES": HYDRA_OVERRIDES,
                        "PYTHONUNBUFFERED": "1"}
    cmd = [*uv, "run", "--frozen", "--extra", "logging", "python", "-m", f"src.{ENTRY}"]
    return subprocess.run(cmd, cwd=root, env=env).returncode


if __name__ == "__main__":
    sys.exit(main())
