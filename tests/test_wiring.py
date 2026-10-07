import torch
from hydra.utils import instantiate

from src.models.fusion.passthrough import PassthroughFusion


def _batch(cfg, b: int = 3) -> dict[str, torch.Tensor]:
    s = cfg.data.image_size
    return {m: torch.randn(b, c, s, s) for m, c in cfg.data.in_channels.items()}


def test_model_interpolates_shapes_from_data(make_cfg) -> None:
    cfg = make_cfg(
        "data.num_classes=7", "data.image_size=8", "data.in_channels={optical: 5, sar: 3}"
    )
    assert cfg.model.num_classes == 7
    assert cfg.model.image_size == 8
    assert dict(cfg.model.in_channels) == {"optical": 5, "sar": 3}


def test_num_classes_change_changes_output_dim_without_code_edit(make_cfg) -> None:
    for k in (3, 11):
        cfg = make_cfg(f"data.num_classes={k}")
        model = instantiate(cfg.model)
        assert model(_batch(cfg)).shape == (3, k)


def test_nested_fusion_group_resolves(make_cfg) -> None:
    cfg = make_cfg("model/fusion=passthrough")
    assert cfg.model.fusion._target_.endswith("PassthroughFusion")
    assert isinstance(instantiate(cfg.model).fusion, PassthroughFusion)


def test_fusion_handles_none_modality() -> None:
    fusion = PassthroughFusion({"optical": 4, "sar": 2})
    out = fusion({"optical": torch.randn(5, 4), "sar": None})
    assert out.shape == (5, 6)
    assert torch.all(out[:, 4:] == 0)


def test_single_modality_passes_through() -> None:
    x = torch.randn(5, 4)
    assert torch.equal(PassthroughFusion({"optical": 4})({"optical": x}), x)
