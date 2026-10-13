# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
"""HF Jobs bootstrap: fetch the repo at the pinned SHA, sync the locked env, run an entrypoint.

Fetched by `hf jobs uv run` from raw GitHub at the same SHA it checks out, so the bootstrap
and the code always match. Env: REPO_SLUG (owner/repo), REPO_SHA, ENTRY (train|evaluate|...),
optional GITHUB_TOKEN for a private repo, optional DATASETS (space-separated Hub dataset
repos). Script args are passed verbatim to the entrypoint.

Each DATASETS repo is downloaded to the job's own disk at <data root>/<owner>/<name> before
the entrypoint starts, and DATA_ROOT points there. Reading shards through a mounted or
streamed Hub repo starved the GPU (~45 samples/s); a local copy downloads at ~1 GB/s.
"""
import json
import os
import subprocess
import sys
import tarfile
import tempfile
import time
import urllib.request
from pathlib import Path


def log(event: str, **fields: object) -> None:
    print(json.dumps({"ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "level": "info",
                      "event": event, "run_id": None, **fields}), flush=True)


def fetch(slug: str, sha: str, dest: Path) -> Path:
    req = urllib.request.Request(f"https://api.github.com/repos/{slug}/tarball/{sha}")
    if token := os.environ.get("GITHUB_TOKEN"):
        req.add_header("Authorization", f"Bearer {token}")
    archive = dest / "repo.tar.gz"
    with urllib.request.urlopen(req, timeout=120) as r, archive.open("wb") as f:
        f.write(r.read())
    with tarfile.open(archive) as tar:
        tar.extractall(dest, filter="data")
    (root,) = [p for p in dest.iterdir() if p.is_dir()]
    return root


def download(root: Path, repos: list[str]) -> Path:
    """Download each dataset repo with the project's pinned hf CLI; return the data root."""
    base = Path("/datasets")
    try:
        base.mkdir(parents=True, exist_ok=True)
    except PermissionError:
        base = Path(tempfile.mkdtemp(prefix="datasets-"))
    for repo in repos:
        t0 = time.monotonic()
        log("dataset_download_start", repo=repo, dest=str(base / repo))
        subprocess.run(["uv", "run", "--frozen", "hf", "download", repo, "--repo-type",
                        "dataset", "--local-dir", str(base / repo)], cwd=root, check=True,
                       stdout=subprocess.DEVNULL)
        size = sum(f.stat().st_size for f in (base / repo).rglob("*") if f.is_file())
        log("dataset_download_done", repo=repo, gb=round(size / 1e9, 1),
            seconds=round(time.monotonic() - t0))
    return base


def main() -> int:
    slug, sha = os.environ["REPO_SLUG"], os.environ["REPO_SHA"]
    entry = os.environ.get("ENTRY", "train")
    overrides = sys.argv[1:]
    log("bootstrap_start", repo=slug, sha=sha, entry=entry, overrides=overrides)
    root = fetch(slug, sha, Path(tempfile.mkdtemp(prefix="repo-")))
    subprocess.run(["uv", "sync", "--frozen", "--extra", "logging"], cwd=root, check=True)
    log("bootstrap_ready", path=str(root))
    env = dict(os.environ)
    if repos := os.environ.get("DATASETS", "").split():
        env["DATA_ROOT"] = str(download(root, repos))
    cmd = ["uv", "run", "--frozen", "--extra", "logging", "python", "-m", f"src.{entry}",
           *overrides]
    return subprocess.run(cmd, cwd=root, env=env).returncode


if __name__ == "__main__":
    sys.exit(main())
