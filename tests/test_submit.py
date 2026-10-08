import json
import os
import shlex
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
OVERRIDES = ["model=dummy_wide", "optim.lr=3e-4", "data.in_channels={optical: 12, sar: 2}"]


def _bash() -> str:
    """Git's bash on Windows, never WSL's System32 bash."""
    exec_path = subprocess.run(["git", "--exec-path"], capture_output=True, text=True).stdout
    for cand in (Path(exec_path.strip()).parents[2] / "bin" / "bash.exe", shutil.which("bash")):
        if cand and Path(cand).exists() and "System32" not in str(cand):
            return str(cand)
    pytest.skip("no usable bash")


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    r = tmp_path / "repo"
    for d in ("scripts", "kernels"):
        shutil.copytree(ROOT / d, r / d)
    git = ["git", "-c", "user.name=t", "-c", "user.email=t@example.com",
           "-c", "core.autocrlf=false"]
    subprocess.run([*git, "init", "-q"], cwd=r, check=True)
    subprocess.run([*git, "add", "-A"], cwd=r, check=True)
    subprocess.run([*git, "commit", "-qm", "init"], cwd=r, check=True)
    subprocess.run(["git", "remote", "add", "origin", "git@github.com:acme/robustness-eval.git"],
                   cwd=r, check=True)
    return r


def _run(repo: Path, script: str, *args: str) -> subprocess.CompletedProcess:
    env = {k: v for k, v in os.environ.items() if k not in ("REPO_SLUG", "KAGGLE_KERNEL")}
    env |= {"HF_BUCKET": "acme/ckpts", "KAGGLE_KERNEL": "acme/rob-train"}
    return subprocess.run([_bash(), f"scripts/{script}", *args], cwd=repo, env=env,
                          capture_output=True, text=True, timeout=120)


def _sha(repo: Path) -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True,
                          text=True, check=True).stdout.strip()


def _hf_commands(stdout: str) -> list[list[str]]:
    return [shlex.split(line) for line in stdout.splitlines() if line.startswith("hf jobs")]


def test_hf_dry_run_passes_every_override_verbatim(repo: Path) -> None:
    res = _run(repo, "submit_hf.sh", "--dry-run", *OVERRIDES)
    assert res.returncode == 0, res.stderr
    (cmd,) = _hf_commands(res.stdout)
    assert cmd[cmd.index("--") + 1:] == OVERRIDES
    sha = _sha(repo)
    raw = "https://raw.githubusercontent.com/acme/robustness-eval"
    assert f"{raw}/{sha}/scripts/hf_entry.py" in cmd
    for flag in (["-s", "HF_TOKEN"], ["-s", "WANDB_API_KEY"],
                 ["-v", "hf://buckets/acme/ckpts/ckpt:/ckpt"],
                 ["--flavor", "a10g-small"], ["--timeout", "4h"]):
        assert any(cmd[i:i + 2] == flag for i in range(len(cmd))), flag
    assert "-d" in cmd


def test_dirty_tree_blocks_and_allow_dirty_bypasses(repo: Path) -> None:
    (repo / "kernels" / "train" / "run.py").write_text("# changed\n", encoding="utf-8")
    for script in ("submit_hf.sh", "submit_kaggle.sh"):
        blocked = _run(repo, script, "--dry-run", "model=dummy")
        assert blocked.returncode != 0 and "dirty" in blocked.stderr, (script, blocked.stderr)
        ok = _run(repo, script, "--dry-run", "--allow-dirty", "model=dummy")
        assert ok.returncode == 0, (script, ok.stderr)


@pytest.mark.parametrize("url", [
    "https://github.com/acme/robustness-eval.git",
    "https://someone@github.com/acme/robustness-eval.git",
    "https://github.com/acme/robustness-eval",
    "ssh://git@github.com/acme/robustness-eval.git",
])
def test_repo_slug_parses_every_github_remote_form(repo: Path, url: str) -> None:
    subprocess.run(["git", "remote", "set-url", "origin", url], cwd=repo, check=True)
    (cmd,) = _hf_commands(_run(repo, "submit_hf.sh", "--dry-run", "model=dummy").stdout)
    assert "REPO_SLUG=acme/robustness-eval" in cmd


def test_sweep_over_2x2_emits_exactly_4_distinct_commands(repo: Path) -> None:
    res = _run(repo, "sweep_hf.sh", "--dry-run", "model=dummy,dummy_wide", "seed=0,1",
               "severities=[0.0,0.5]")
    assert res.returncode == 0, res.stderr
    cmds = _hf_commands(res.stdout)
    tails = {tuple(c[c.index("--") + 1:]) for c in cmds}
    assert len(cmds) == 4 and len(tails) == 4
    assert {(t[0], t[1]) for t in tails} == {
        ("model=dummy", "seed=0"), ("model=dummy", "seed=1"),
        ("model=dummy_wide", "seed=0"), ("model=dummy_wide", "seed=1"),
    }
    assert all(t[2] == "severities=[0.0,0.5]" for t in tails)


def test_both_scripts_set_run_env(repo: Path) -> None:
    hf = _run(repo, "submit_hf.sh", "--dry-run", "model=dummy")
    (cmd,) = _hf_commands(hf.stdout)
    assert "RUN_ENV=hf_jobs" in cmd
    kg = _run(repo, "submit_kaggle.sh", "--dry-run", "model=dummy")
    assert kg.returncode == 0, kg.stderr
    assert "RUN_ENV=kaggle" in kg.stdout


def test_kaggle_dry_run_stamps_kernel_and_overrides_round_trip(repo: Path) -> None:
    res = _run(repo, "submit_kaggle.sh", "--dry-run", *OVERRIDES)
    assert res.returncode == 0, res.stderr
    assert "kaggle kernels push -p" in res.stdout
    build = repo / "outputs" / "kaggle_build" / "acme_rob-train"
    ns: dict = {}
    src = (build / "run.py").read_text()
    exec(compile(src.split("\n\ndef main")[0], "run.py", "exec"), ns)
    assert ns["REPO_SLUG"] == "acme/robustness-eval" and ns["REPO_SHA"] == _sha(repo)
    assert shlex.split(ns["HYDRA_OVERRIDES"]) == OVERRIDES
    meta = json.loads((build / "kernel-metadata.json").read_text())
    assert meta["id"] == "acme/rob-train" and meta["kernel_type"] == "script"
    assert meta["enable_gpu"] is True and meta["enable_internet"] is True
