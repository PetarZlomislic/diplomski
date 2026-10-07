import pytest
import torch
from hydra.utils import instantiate

from tests.conftest import group_names, model_names


@pytest.mark.parametrize("model_name", model_names())
@pytest.mark.parametrize("dropped", ["optical", "sar"])
def test_every_model_accepts_a_none_modality(make_cfg, model_name: str, dropped: str) -> None:
    cfg = make_cfg(f"model={model_name}")
    model = instantiate(cfg.model)
    s = cfg.data.image_size
    batch = {m: torch.randn(2, c, s, s) for m, c in cfg.data.in_channels.items()}
    batch[dropped] = None
    assert model(batch).shape == (2, cfg.data.num_classes)


def _deg_batch() -> dict[str, torch.Tensor | None]:
    g = torch.Generator().manual_seed(0)
    return {
        "optical": torch.randn(2, 12, 8, 8, generator=g),
        "sar": torch.randn(2, 2, 8, 8, generator=g),
        "label": torch.ones(2, 3),
    }


@pytest.mark.parametrize("deg_name", group_names("degradation"))
def test_every_degradation_is_exact_identity_at_severity_zero(make_cfg, deg_name: str) -> None:
    deg = instantiate(make_cfg(f"degradation={deg_name}").degradation)
    batch = _deg_batch()
    out = deg(batch, 0.0, torch.Generator().manual_seed(1))
    assert out.keys() == batch.keys()
    assert all(torch.equal(out[k], batch[k]) for k in batch)


@pytest.mark.parametrize("deg_name", [d for d in group_names("degradation") if d != "none"])
def test_degradation_at_severity_one_changes_tensor(make_cfg, deg_name: str) -> None:
    deg = instantiate(make_cfg(f"degradation={deg_name}").degradation)
    batch = _deg_batch()
    out = deg(batch, 1.0, torch.Generator().manual_seed(1))
    assert any(not torch.equal(out[k], batch[k]) for k in ("optical", "sar"))
