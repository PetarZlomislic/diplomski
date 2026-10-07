import pytest
import torch
from hydra.utils import instantiate

from tests.conftest import model_names


@pytest.mark.parametrize("model_name", model_names())
@pytest.mark.parametrize("dropped", ["optical", "sar"])
def test_every_model_accepts_a_none_modality(make_cfg, model_name: str, dropped: str) -> None:
    cfg = make_cfg(f"model={model_name}")
    model = instantiate(cfg.model)
    s = cfg.data.image_size
    batch = {m: torch.randn(2, c, s, s) for m, c in cfg.data.in_channels.items()}
    batch[dropped] = None
    assert model(batch).shape == (2, cfg.data.num_classes)
