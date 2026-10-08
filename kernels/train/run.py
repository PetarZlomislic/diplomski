"""Kaggle kernel shim: fetch the repo at a pinned SHA, install pinned deps, call src.train.main().

scripts/submit_kaggle.sh stamps the constants below into a build copy of this file; Kaggle
pushes a single code file and has no per-run environment, so they travel inside it.
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


def main() -> None:
    os.environ["RUN_ENV"] = "kaggle"
    os.environ["HYDRA_OVERRIDES"] = HYDRA_OVERRIDES
    dest = Path("/kaggle/tmp") if Path("/kaggle").exists() else Path.cwd() / "kaggle_tmp"
    dest.mkdir(parents=True, exist_ok=True)
    archive = dest / "repo.tar.gz"
    req = urllib.request.Request(f"https://api.github.com/repos/{REPO_SLUG}/tarball/{REPO_SHA}")
    with urllib.request.urlopen(req, timeout=120) as r, archive.open("wb") as f:
        f.write(r.read())
    with tarfile.open(archive) as tar:
        tar.extractall(dest, filter="data")
    root = next(p for p in dest.iterdir() if p.is_dir() and p.name.endswith(REPO_SHA[:7]))

    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "uv"], check=True)
    subprocess.run([sys.executable, "-m", "uv", "pip", "install", "--system", "--python",
                    sys.executable, "-r", str(root / "pyproject.toml"), "--extra", "logging"],
                   check=True)

    sys.path.insert(0, str(root))
    os.chdir(root)
    sys.argv = [f"src.{ENTRY}"]
    if ENTRY == "evaluate":
        from src.evaluate import main as entry_main
    else:
        from src.train import main as entry_main
    entry_main()


if __name__ == "__main__":
    main()
