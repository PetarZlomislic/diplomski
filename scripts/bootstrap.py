# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
"""Remote job bootstrap: fetch the repo at a pinned SHA, sync the LOCKED env, run an entrypoint.

HF Jobs: fetched by `hf jobs uv run` from raw GitHub at the same SHA it checks out, so the
bootstrap and the code always match; settings come from the job env, script args are the
Hydra overrides. Kaggle pushes a single code file and has no per-run environment, so
scripts/submit.py stamps the settings into STAMPED in a build copy of this file.

Settings: REPO_SLUG (owner/repo), REPO_SHA, ENTRY (train|evaluate|...), DATASETS
(space-separated Hub dataset repos), MIN_FREE_GB (disk they need), optional GITHUB_TOKEN for
a private repo, and on Kaggle HYDRA_OVERRIDES. Each DATASETS repo is downloaded to the job's
own disk at <data root>/<owner>/<name> before the entrypoint starts, and DATA_ROOT points
there: reading shards through a mounted or streamed Hub repo starved the GPU (~45 samples/s).
On Kaggle that is the scratch disk, not the 20 GB /kaggle/working output area; a private repo
there needs a Kaggle Secret named HF_TOKEN attached to the kernel.

The project runs in its own uv venv, never on top of preinstalled packages (Kaggle's include
extras, e.g. transformers, that conflict with the pinned versions once imported).
"""
import json
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
import urllib.request
from pathlib import Path

STAMPED: dict[str, str] = {}  # filled in by scripts/submit.py for Kaggle
KAGGLE = Path("/kaggle")


def log(event: str, **fields: object) -> None:
    print(json.dumps({"ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "level": "info",
                      "event": event, "run_id": None, **fields}), flush=True)


def setting(name: str, default: str = "") -> str:
    return STAMPED.get(name) or os.environ.get(name) or default


def hf_token() -> str | None:
    if token := os.environ.get("HF_TOKEN"):
        return token
    try:
        from kaggle_secrets import UserSecretsClient

        return UserSecretsClient().get_secret("HF_TOKEN")
    except Exception:  # not attached, or not on Kaggle
        return None


def fetch(slug: str, sha: str, dest: Path) -> Path:
    req = urllib.request.Request(f"https://api.github.com/repos/{slug}/tarball/{sha}")
    if token := os.environ.get("GITHUB_TOKEN"):
        req.add_header("Authorization", f"Bearer {token}")
    archive = dest / "repo.tar.gz"
    with urllib.request.urlopen(req, timeout=120) as r, archive.open("wb") as f:
        f.write(r.read())
    with tarfile.open(archive) as tar:
        tar.extractall(dest, filter="data")
    return next(p for p in dest.iterdir() if p.is_dir() and p.name.endswith(sha[:7]))


def download(uv: list[str], root: Path, base: Path, repos: list[str]) -> Path:
    """Download each dataset repo with the project's pinned hf CLI; return the data root."""
    try:
        base.mkdir(parents=True, exist_ok=True)
    except PermissionError:
        base = Path(tempfile.mkdtemp(prefix="datasets-"))
    # MIN_FREE_GB: set by `submit.py auto` from the datasets' size; Kaggle's default covers
    # the 41.5 GB pilot tier.
    need = float(setting("MIN_FREE_GB", "50" if STAMPED else "0"))
    free = shutil.disk_usage(base).free / 1e9
    if free < need:
        raise SystemExit(f"only {free:.0f} GB free at {base}; need >= {need:.0f} GB")
    env = dict(os.environ)
    if token := hf_token():
        env["HF_TOKEN"] = token
    for repo in repos:
        t0 = time.monotonic()
        log("dataset_download_start", repo=repo, dest=str(base / repo))
        subprocess.run([*uv, "run", "--frozen", "hf", "download", repo, "--repo-type",
                        "dataset", "--local-dir", str(base / repo)], cwd=root, env=env,
                       check=True, stdout=subprocess.DEVNULL)
        size = sum(f.stat().st_size for f in (base / repo).rglob("*") if f.is_file())
        log("dataset_download_done", repo=repo, gb=round(size / 1e9, 1),
            seconds=round(time.monotonic() - t0))
    return base


def main() -> int:
    slug, sha, entry = setting("REPO_SLUG"), setting("REPO_SHA"), setting("ENTRY", "train")
    overrides = sys.argv[1:]
    log("bootstrap_start", repo=slug, sha=sha, entry=entry,
        overrides=overrides or setting("HYDRA_OVERRIDES"))
    dest = KAGGLE / "tmp" if KAGGLE.exists() else Path(tempfile.mkdtemp(prefix="repo-"))
    dest.mkdir(parents=True, exist_ok=True)
    root = fetch(slug, sha, dest)
    if shutil.which("uv"):
        uv = ["uv"]
    else:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "uv"], check=True)
        uv = [sys.executable, "-m", "uv"]
    subprocess.run([*uv, "sync", "--frozen", "--extra", "logging"], cwd=root, check=True)
    log("bootstrap_ready", path=str(root))

    env = os.environ | {"REPO_SHA": sha, "PYTHONUNBUFFERED": "1"}
    if STAMPED:
        env |= {"RUN_ENV": "kaggle", "HYDRA_OVERRIDES": setting("HYDRA_OVERRIDES")}
    if repos := setting("DATASETS").split():
        base = dest / "datasets" if KAGGLE.exists() else Path("/datasets")
        env["DATA_ROOT"] = str(download(uv, root, base, repos))
    cmd = [*uv, "run", "--frozen", "--extra", "logging", "python", "-m", f"src.{entry}",
           *overrides]
    return subprocess.run(cmd, cwd=root, env=env).returncode


if __name__ == "__main__":
    sys.exit(main())
