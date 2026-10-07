from collections.abc import Callable
from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig

ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = ROOT / "configs"


def compose_cfg(*overrides: str, config_name: str = "config") -> DictConfig:
    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base="1.3"):
        return compose(config_name=config_name, overrides=list(overrides))


@pytest.fixture
def make_cfg() -> Callable[..., DictConfig]:
    return compose_cfg


def group_names(group: str) -> list[str]:
    return sorted(p.stem for p in (CONFIG_DIR / group).glob("*.yaml"))


def model_names() -> list[str]:
    return group_names("model")


@pytest.fixture(scope="session")
def trained_run(tmp_path_factory: pytest.TempPathFactory) -> Callable[[str], Path]:
    """Train a smoke model in-process once per model name; return its run dir."""
    from src.train import run

    cache: dict[str, Path] = {}

    def _train(model: str = "dummy") -> Path:
        if model not in cache:
            out = tmp_path_factory.mktemp(f"run_{model}")
            cfg = compose_cfg(f"model={model}", "+experiment=smoke", f"output_dir={out}")
            cache[model] = run(cfg, {"model": model, "data": "dummy"})
        return cache[model]

    return _train
