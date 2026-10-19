"""Experiment 1 components: arms, degradations, data preparation, pre-flight."""
import json
from pathlib import Path

import pytest
import torch
from hydra.utils import instantiate

from src.degradations import (
    CloudOcclusion,
    ModalityDropout,
    Speckle,
    TemporalMisalignment,
    cloud_mask,
    looks,
)
from tests.conftest import compose_cfg

ARMS = ("opt", "sar", "early", "late", "gate_global", "gate_spatial")


def _batch(b: int = 4, s: int = 16) -> dict[str, torch.Tensor]:
    g = torch.Generator().manual_seed(0)
    return {"optical": torch.randn(b, 12, s, s, generator=g),
            "sar": torch.randn(b, 2, s, s, generator=g),
            "sar_shift_2": torch.randn(b, 2, s, s, generator=g),
            "label": torch.ones(b, 19)}


def _arm(name: str, seed: int = 0):
    torch.manual_seed(seed)
    return instantiate(compose_cfg(f"model={name}", "data.image_size=16").model)


# ------------------------------------------------------------------ arms

def test_all_arms_share_encoder_initialisation() -> None:
    opt, sar, early, late, gate = (_arm(a, seed=7) for a in
                                   ("opt", "sar", "early", "late", "gate_spatial"))
    stem = early.encoder.stem[0].weight
    assert torch.equal(opt.encoder.stem[0].weight, stem[:, :12])
    assert torch.equal(sar.encoder.stem[0].weight, stem[:, 12:])
    for m, sl in (("optical", slice(0, 12)), ("sar", slice(12, 14))):
        assert torch.equal(late.encoders[m].stem[0].weight, stem[:, sl])
        assert torch.equal(gate.encoders[m].stem[0].weight, stem[:, sl])
    deep = lambda e: e.blocks[-1].conv2.weight  # noqa: E731
    assert torch.equal(deep(opt.encoder), deep(gate.encoders["sar"]))
    assert not torch.equal(stem, _arm("early", seed=8).encoder.stem[0].weight)


def test_gate_arms_are_parameter_matched() -> None:
    count = lambda m: sum(p.numel() for p in m.parameters())  # noqa: E731
    assert count(_arm("gate_global")) == count(_arm("gate_spatial"))


@pytest.mark.parametrize("arm", ARMS)
def test_arm_survives_every_dropout(arm: str) -> None:
    model = _arm(arm).eval()
    for drop in ("optical", "sar"):
        b = _batch()
        b[drop] = None
        assert model(b).shape == (4, 19)


@pytest.mark.parametrize("arm", ["late", "gate_global", "gate_spatial"])
def test_dropped_modality_gets_zero_weight(arm: str) -> None:
    model = _arm(arm).eval()
    model(_batch())
    w = model.gate_weights
    assert w.shape == (4, 2) and torch.allclose(w.sum(1), torch.ones(4))
    b = _batch()
    b["sar"] = None
    model(b)
    assert torch.equal(model.gate_weights[:, 1], torch.zeros(4))


def test_unimodal_arm_ignores_the_other_modality() -> None:
    model = _arm("opt").eval()
    b = _batch()
    with torch.no_grad():
        ref = model(b)
        b["sar"] = b["sar"] * 100
        assert torch.equal(model(b), ref)


# ------------------------------------------------------------------ degradations

@pytest.mark.parametrize("fraction", [0.0, 0.25, 0.5, 0.75, 1.0])
def test_cloud_mask_fraction_is_exact(fraction: float) -> None:
    m = cloud_mask(3, 40, 40, fraction, torch.Generator().manual_seed(0))
    assert (m.sum(dim=(1, 2)) == round(fraction * 1600)).all()


def test_cloud_mask_is_spatially_coherent() -> None:
    """Blobs, not salt and pepper: masked pixels mostly have masked neighbours."""
    m = cloud_mask(4, 64, 64, 0.3, torch.Generator().manual_seed(0))
    same = (m[:, 1:, :] == m[:, :-1, :]).float().mean()
    assert same > 0.9  # independent pixels would give 0.58


def test_cloud_only_touches_optical_and_fills_with_mean() -> None:
    b = _batch()
    out = CloudOcclusion()(b, 0.5, torch.Generator().manual_seed(0))
    assert torch.equal(out["sar"], b["sar"])
    changed = (out["optical"] != b["optical"]).any(1)
    assert changed.float().mean() == pytest.approx(0.5, abs=0.01)
    assert (out["optical"].permute(1, 0, 2, 3)[:, changed] == 0).all()


def test_speckle_looks_mapping() -> None:
    assert [looks(s, 1.0) for s in (0.25, 0.5, 1.0)] == [4.0, 2.0, 1.0]


def test_speckle_db_noise_has_gamma_statistics() -> None:
    x = {"sar": torch.zeros(8, 2, 64, 64), "optical": torch.zeros(8, 12, 64, 64)}
    out = Speckle(scale="db")(x, 1.0, torch.Generator().manual_seed(0))
    assert torch.equal(out["optical"], x["optical"])
    lin = 10 ** (out["sar"] / 10)  # back to linear intensity: n ~ Gamma(1, 1)
    assert lin.mean().item() == pytest.approx(1.0, abs=0.02)
    assert lin.var().item() == pytest.approx(1.0, abs=0.05)


def test_speckle_uses_band_std_in_standardised_space() -> None:
    class DM:
        band_stats = {"sar": {"mean": [-12.0, -19.0], "std": [4.0, 2.0]}}

    sp = Speckle(scale="db")
    sp.bind(DM())
    x = {"sar": torch.zeros(4, 2, 32, 32)}
    a = sp(x, 1.0, torch.Generator().manual_seed(0))["sar"]
    raw = Speckle(scale="db")(x, 1.0, torch.Generator().manual_seed(0))["sar"]
    assert torch.allclose(a[:, 0] * 4.0, raw[:, 0]) and torch.allclose(a[:, 1] * 2.0, raw[:, 1])


def test_dropout_and_temporal() -> None:
    b = _batch()
    assert ModalityDropout("sar")(b, 1.0, torch.Generator())["sar"] is None
    out = TemporalMisalignment()(b, 2, torch.Generator())
    assert torch.equal(out["sar"], b["sar_shift_2"]) and torch.equal(out["optical"],
                                                                      b["optical"])
    with pytest.raises(KeyError, match="sar_shift_4"):
        TemporalMisalignment()(b, 4, torch.Generator())


# ------------------------------------------------------------------ data preparation

def test_temporal_neighbours_are_exact_steps_away() -> None:
    from src.prepare_data import attach_shifts, grid_position, synthetic_source

    recs = synthetic_source(n=400, seed=0)
    by_id = {r["patch_id"]: r for r in recs}
    test = [r for r in recs if r["split"] == "test"]
    out, counts = attach_shifts(test, recs, [1, 2])
    for k in (1, 2):
        assert counts[str(k)]["exact"] == len(test)
    for r in out:
        tile, row, col = grid_position(r["patch_id"])
        nb = next(p for p in by_id.values() if p["sar"] is r["sar_shift_2"])
        t2, r2, c2 = grid_position(nb["patch_id"])
        assert t2 == tile and max(abs(r2 - row), abs(c2 - col)) == 2
    assert all("sar_shift_1" not in r for r in test)  # inputs are not mutated


def test_prepared_tier_has_stats_and_test_only_shifts(tmp_path: Path) -> None:
    from litdata import StreamingDataset

    from src.prepare_data import prepare, synthetic_source

    prepare(synthetic_source(n=600, seed=0), tmp_path, [0.5], temporal_steps=[1, 4])
    tier = tmp_path / "frac0.5"
    manifest = json.loads((tier / "manifest.json").read_text())
    assert len(manifest["band_stats"]["optical"]["mean"]) == 12
    assert manifest["temporal"]["steps"] == [1, 4]
    assert {"sar_shift_1", "sar_shift_4"} <= set(StreamingDataset(str(tier / "test"))[0])
    assert "sar_shift_1" not in StreamingDataset(str(tier / "train"))[0]


def test_streaming_standardises_with_manifest_stats(tmp_path: Path) -> None:
    from src.prepare_data import prepare, synthetic_source

    recs = synthetic_source(n=600, seed=0)
    for r in recs:  # shift optical far from N(0, 1) so standardisation is visible
        r["optical"] = r["optical"] * 50 + 1000
    prepare(recs, tmp_path, [1.0], temporal_steps=[1])
    cfg = compose_cfg("data=streaming", f"data.data_uri={(tmp_path / 'full').as_posix()}",
                      "data.image_size=8", "data.batch_size=64",
                      f"data.cache_dir={(tmp_path / 'cache').as_posix()}")
    dm = instantiate(cfg.data)
    dm.setup("fit")
    raw = next(iter(dm.train_dataloader()))
    assert raw["optical"].mean() > 500  # workers yield raw values; prepare_batch standardises
    x = torch.cat([dm.prepare_batch(b)["optical"] for b in dm.train_dataloader()])
    assert abs(x.mean().item()) < 0.05 and abs(x.std().item() - 1) < 0.05
    t = dm.prepare_batch(next(iter(dm.test_dataloader())))
    assert abs(t["sar_shift_1"].std().item() - 1) < 0.2


def test_train_flips_are_per_sample_and_shared_across_modalities() -> None:
    from src.data import BaseDataModule

    dm = BaseDataModule(2, {"optical": 3, "sar": 1}, 4, batch_size=64, augment=True)
    torch.manual_seed(0)
    img = torch.arange(16.0).view(1, 1, 4, 4).repeat(64, 1, 1, 1)
    out = dm.prepare_batch({"optical": img.repeat(1, 3, 1, 1), "sar": img.clone(),
                            "label": torch.ones(64, 2, dtype=torch.uint8)}, train=True)
    assert out["label"].dtype == torch.float32
    assert torch.equal(out["optical"][:, :1], out["sar"])  # same flip for both modalities
    flipped = (out["sar"] != img).flatten(1).any(1)
    assert 0 < flipped.sum() < 64  # per sample, not one coin for the whole batch
    assert torch.equal(dm.prepare_batch({"sar": img}, train=False)["sar"], img)


# ------------------------------------------------------------------ end to end

@pytest.fixture(scope="module")
def pilot(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """opt, sar and gate_spatial trained on dummy data, each swept over every degradation."""
    from src.evaluate import evaluate
    from src.train import run

    root = tmp_path_factory.mktemp("pilot")
    grids = {"cloud": [0.5, 1.0], "speckle": [0.5, 1.0], "drop_optical": [1.0],
             "drop_sar": [1.0], "temporal": [1, 2]}
    for arm in ("opt", "sar", "gate_spatial"):
        cfg = compose_cfg(f"model={arm}", "+experiment=smoke", "data.num_classes=19",
                          "data.n_samples=40", "model.encoder.width=8",
                          f"output_dir={root.as_posix()}")
        run_dir = run(cfg, {"model": arm, "data": "dummy"})
        for deg, sev in grids.items():
            evaluate(str(run_dir), "best", compose_cfg(f"degradation={deg}").degradation,
                     deg, sev, 12345)
    return root


def test_pilot_rows_carry_per_class_flops_and_weights(pilot: Path) -> None:
    from src.analysis import load_results

    df = load_results(pilot)
    assert all(len(json.loads(x)) == 19 for x in df["f1_per_class"])
    assert (df["flops"] > 0).all()
    gate = df[(df["model"] == "gate_spatial") & (df["degradation"] == "drop_sar")]
    assert json.loads(gate["modality_weights"].iloc[0])["sar"] == 0.0
    assert df[df["model"] == "opt"]["modality_weights"].isna().all()


def test_preflight_wiring_checks_pass_on_pilot(pilot: Path) -> None:
    from src.preflight import preflight

    res = preflight(pilot, refs={}, tol=0.05, n_batches=2)
    for name in ("1 identity", "2 seed", "3 leakage"):
        assert res[name][0] == "PASS", (name, res[name])
    assert res["4 reference"][0] == "SKIP"


def test_preflight_catches_a_leaking_degradation(pilot: Path, monkeypatch) -> None:
    import src.preflight as pf
    from src.analysis import load_results

    df = load_results(pilot)
    rows = (df["model"] == "opt") & (df["degradation"] == "speckle")
    df.loc[rows, "f1_macro"] = df.loc[rows, "f1_macro"] + 0.01
    status, detail = pf.check_leakage(df)
    assert status == "FAIL" and "opt/speckle" in detail


def test_analysis_runs_on_pilot(pilot: Path) -> None:
    from src.analysis import analyse

    files = analyse(pilot, pilot / "analysis")
    assert files["fig2"].exists() and files["table2"].exists()


def test_sweep_eval_runs_every_degradation_of_the_stage(trained_run) -> None:
    from src.analysis import load_results
    from src.evaluate import sweep

    run_dir = trained_run("dummy")
    cfg = compose_cfg(f"run_dir={run_dir.as_posix()}", "stage=pilot",
                      config_name="evaluate")
    assert sweep(cfg) == sum(len(v) for v in cfg.grids.pilot.values())
    got = load_results(run_dir).groupby("degradation")["severity"].apply(sorted).to_dict()
    for deg, sev in cfg.grids.pilot.items():
        assert got[deg] == [float(s) for s in sev]


def test_preflight_identity_compares_prepared_batches() -> None:
    """With real band stats, s=0 must equal the PREPARED clean batch (not the raw one)."""
    from src.data import DummyDataModule
    from src.preflight import check_identity

    dm = DummyDataModule(num_classes=3, in_channels={"optical": 12, "sar": 2}, image_size=8,
                         n_samples=20)
    dm.band_stats = {"optical": {"mean": [5.0] * 12, "std": [2.0] * 12},
                     "sar": {"mean": [-12.0, -19.0], "std": [5.0, 5.0]}}
    dm.setup("test")
    import pandas as pd

    empty = pd.DataFrame(columns=["run_id", "degradation", "severity", "f1_macro"])
    status, detail = check_identity(dm, ["cloud", "speckle", "temporal"], 2, empty)
    assert status == "PASS", detail


def test_exp1_run_trains_and_sweeps_every_model(tmp_path: Path) -> None:
    from src.analysis import load_results
    from src.exp1_run import main

    status = main(["models=[dummy,dummy_wide]", "seeds=[0]", "stage=pilot",
                   f"train=[+experiment=smoke,output_dir={tmp_path.as_posix()}]"])
    assert status == 0
    df = load_results(tmp_path)
    assert set(df["model"]) == {"dummy", "dummy_wide"}
    assert {"none", "cloud", "speckle", "drop_optical", "drop_sar", "temporal"} <= set(
        df["degradation"])


def test_exp1_run_continues_after_a_failed_arm(tmp_path: Path) -> None:
    from src.exp1_run import main

    status = main(["models=[no_such_model,dummy]", "stage=none",
                   f"train=[+experiment=smoke,output_dir={tmp_path.as_posix()}]"])
    assert status == 1
    assert len(list(tmp_path.glob("*/checkpoints/best.ckpt"))) == 1
