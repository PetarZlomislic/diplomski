import re
import subprocess
import sys
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from src.env import Env, Runtime, detect
from src.obs import resolve_wandb_mode

ROOT = Path(__file__).resolve().parent.parent
ENV_VARS = ["RUN_ENV", "KAGGLE_KERNEL_RUN_TYPE", "HAS_INTERNET", "OUTPUT_ROOT", "DATA_ROOT",
            "MAX_RUNTIME_S", "HYDRA_OVERRIDES"]


@pytest.fixture(autouse=True)
def clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for v in ENV_VARS:
        monkeypatch.delenv(v, raising=False)


def test_run_env_wins_over_autodetection(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("KAGGLE_KERNEL_RUN_TYPE", "Batch")
    monkeypatch.setenv("RUN_ENV", "hf_jobs")
    assert detect().runtime is Runtime.HF_JOBS
    monkeypatch.setenv("RUN_ENV", "local")
    assert detect().runtime is Runtime.LOCAL


def test_kaggle_markers_detect_kaggle(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("KAGGLE_KERNEL_RUN_TYPE", "Batch")
    monkeypatch.setenv("HAS_INTERNET", "0")
    env = detect()
    assert env.runtime is Runtime.KAGGLE
    assert env.max_runtime_s == 9 * 3600
    assert not env.has_internet


def test_no_markers_is_local_with_outputs_root() -> None:
    env = detect()
    assert env.runtime is Runtime.LOCAL
    assert env.output_root == Path("outputs").resolve()


def test_hf_jobs_root_is_bucket_mount(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RUN_ENV", "hf_jobs")
    assert detect().output_root == Path("/ckpt")


def test_secret_returns_none_when_absent(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SURELY_NOT_A_SECRET", raising=False)
    for runtime in Runtime:
        env = Env(runtime, Path("."), "", True, None)
        assert env.secret("SURELY_NOT_A_SECRET") is None


def test_hydra_argv_reads_overrides_env_only_on_kaggle(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HYDRA_OVERRIDES", "model=dummy_wide 'optim.lr=0.01'")
    kaggle = Env(Runtime.KAGGLE, Path("."), "", True, None)
    local = Env(Runtime.LOCAL, Path("."), "", True, None)
    assert kaggle.hydra_argv(["prog", "ignored=1"]) == ["prog", "model=dummy_wide", "optim.lr=0.01"]
    assert local.hydra_argv(["prog", "model=dummy"]) == ["prog", "model=dummy"]


def test_train_on_kaggle_reads_overrides_from_env(tmp_path: Path) -> None:
    out = tmp_path.as_posix()
    env = {k: v for k, v in __import__("os").environ.items() if k not in ENV_VARS}
    env |= {
        "RUN_ENV": "kaggle",
        "HAS_INTERNET": "0",
        "HYDRA_OVERRIDES": f"model=dummy_wide +experiment=smoke output_dir={out}",
    }
    result = subprocess.run(
        [sys.executable, "-m", "src.train", "model=dummy"],  # CLI arg must be ignored
        cwd=ROOT, env=env, capture_output=True, text=True, timeout=300,
    )
    assert result.returncode == 0, result.stderr[-2000:]
    (run_cfg,) = tmp_path.glob("*/run_config.yaml")
    assert OmegaConf.load(run_cfg).cfg.model.hidden_dim == 128  # dummy_wide


def test_wandb_forced_offline_without_internet() -> None:
    assert resolve_wandb_mode("online", has_internet=False) == "offline"
    assert resolve_wandb_mode("online", has_internet=True) == "online"


def test_no_module_outside_env_mentions_runtime_specifics() -> None:
    pattern = re.compile(r"kaggle|/ckpt", re.IGNORECASE)
    offenders = [
        f"{p.relative_to(ROOT)}:{i}"
        for p in (ROOT / "src").rglob("*.py")
        if p.name != "env.py"
        for i, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1)
        if pattern.search(line)
    ]
    assert not offenders, offenders
