from collections.abc import Callable
from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig

ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = ROOT / "configs"


@pytest.fixture
def make_cfg() -> Callable[..., DictConfig]:
    def _make(*overrides: str) -> DictConfig:
        with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base="1.3"):
            return compose(config_name="config", overrides=list(overrides))

    return _make


def model_names() -> list[str]:
    return sorted(p.stem for p in (CONFIG_DIR / "model").glob("*.yaml"))
