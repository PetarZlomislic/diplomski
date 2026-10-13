"""Kaggle kernel shim: fetch the repo at a pinned SHA, sync the LOCKED env, run an entrypoint.

scripts/submit_kaggle.sh stamps the constants below into a build copy of this file; Kaggle
pushes a single code file and has no per-run environment, so they travel inside it.

The project runs in its own uv venv, not on top of Kaggle's preinstalled packages: those
include extras (e.g. transformers) that conflict with the pinned versions once imported.

DATASETS (space-separated Hub dataset repos) are downloaded to Kaggle's scratch disk, NOT to
/kaggle/working (the 20 GB output area), and DATA_ROOT points there. A private repo needs an
HF token: add a Kaggle Secret named HF_TOKEN and attach it to this kernel (Add-ons -> Secrets).
"""
import os
import shutil
import subprocess
import sys
import tarfile
import urllib.request
from pathlib import Path

REPO_SLUG = "__REPO_SLUG__"
REPO_SHA = "__REPO_SHA__"
HYDRA_OVERRIDES = "__HYDRA_OVERRIDES__"
ENTRY = "__ENTRY__"
DATASETS = "__DATASETS__"
MIN_FREE_GB = 50  # the pilot tier is 41.5 GB


def log(msg: str) -> None:
    print(f"[kaggle-shim] {msg}", flush=True)


def hf_token() -> str | None:
    try:
        from kaggle_secrets import UserSecretsClient

        return UserSecretsClient().get_secret("HF_TOKEN")
    except Exception as e:  # not attached, or not on Kaggle
        log(f"no HF_TOKEN Kaggle secret ({type(e).__name__}); downloading anonymously")
        return os.environ.get("HF_TOKEN")


def download(uv: list[str], root: Path, base: Path, repos: list[str]) -> dict[str, str]:
    base.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(base).free / 1e9
    log(f"{free:.0f} GB free at {base}")
    if free < MIN_FREE_GB:
        raise SystemExit(f"only {free:.0f} GB free at {base}; need >= {MIN_FREE_GB} GB")
    env = dict(os.environ)
    if token := hf_token():
        env["HF_TOKEN"] = token
    for repo in repos:
        log(f"downloading {repo} -> {base / repo}")
        subprocess.run([*uv, "run", "--frozen", "hf", "download", repo, "--repo-type",
                        "dataset", "--local-dir", str(base / repo)], cwd=root, env=env,
                       check=True, stdout=subprocess.DEVNULL)
        size = sum(f.stat().st_size for f in (base / repo).rglob("*") if f.is_file()) / 1e9
        log(f"downloaded {repo}: {size:.1f} GB")
    return {"DATA_ROOT": str(base)}


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
                        "REPO_SHA": REPO_SHA, "PYTHONUNBUFFERED": "1"}
    if repos := DATASETS.split():
        env |= download(uv, root, dest / "datasets", repos)
    cmd = [*uv, "run", "--frozen", "--extra", "logging", "python", "-m", f"src.{ENTRY}"]
    return subprocess.run(cmd, cwd=root, env=env).returncode


if __name__ == "__main__":
    sys.exit(main())
