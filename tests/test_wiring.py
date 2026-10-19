import torch
from hydra.utils import instantiate

from src.models import DummyModel


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
    cfg = make_cfg("model=late", "model/fusion=gate_global")
    assert cfg.model.fusion._target_.endswith("GlobalGateFusion")
    assert type(instantiate(cfg.model).fusion).__name__ == "GlobalGateFusion"


def test_dummy_zero_fills_a_dropped_modality() -> None:
    model = DummyModel(3, {"optical": 4, "sar": 2}, 8)
    x = torch.randn(5, 4, 8, 8)
    want = model.head(torch.cat([x.mean(dim=(2, 3)), torch.zeros(5, 2)], 1))
    assert torch.equal(model({"optical": x, "sar": None}), want)


def test_evaluate_reconstructs_model_without_model_override(trained_run, make_cfg) -> None:
    from src.evaluate import resolve_ckpt
    from src.models import LitModule

    run_dir = trained_run("dummy_wide")
    ckpt_path, _, _ = resolve_ckpt("best", str(run_dir))
    loaded = LitModule.load_from_checkpoint(ckpt_path, map_location="cpu").model
    expected = instantiate(make_cfg("model=dummy_wide", "+experiment=smoke").model)
    got = {k: v.shape for k, v in loaded.state_dict().items()}
    want = {k: v.shape for k, v in expected.state_dict().items()}
    assert got == want
    assert type(loaded) is type(expected)


def test_checkpoint_selection_defaults_to_best_and_is_recorded(trained_run, make_cfg) -> None:
    from src.evaluate import evaluate

    eval_cfg = make_cfg(config_name="evaluate")
    assert eval_cfg.ckpt == "best"
    run_dir = trained_run("dummy")
    (row,) = evaluate(
        str(run_dir), eval_cfg.ckpt, eval_cfg.degradation, "none", [0.0], eval_cfg.degradation_seed
    )
    assert row.ckpt_kind == "best"
