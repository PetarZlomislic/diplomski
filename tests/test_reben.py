"""reBEN loader against a tiny fake LMDB with the real layout (native band resolutions,
SafeTensors values, metadata.parquet). No real data is read."""
from pathlib import Path

import lmdb
import numpy as np
import pandas as pd
import pytest
from litdata import StreamingDataset
from safetensors.numpy import save as st_save

from src.data import reben

NATIVE = {b: 120 for b in ("B02", "B03", "B04", "B08")}
NATIVE.update({b: 60 for b in ("B05", "B06", "B07", "B8A", "B11", "B12")})
NATIVE.update({"B01": 20, "B09": 20})


def _fake_reben(root: Path, tiles: int = 3, grid: int = 6) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    env = lmdb.open(str(root / reben.LMDB_DIR), map_size=2**30)
    rows = []
    with env.begin(write=True) as txn:
        for t in range(tiles):
            for h in range(grid):
                for v in range(grid):
                    pid = f"S2A_MSIL2A_20170613T101031_N9999_R022_T33UU{t}_{h:02d}_{v:02d}"
                    s1 = f"S1B_IW_GRDH_1SDV_20170612T165809_33UU{t}_{h:02d}_{v:02d}"
                    s2 = {b: rng.integers(0, 10000, (n, n), dtype=np.uint16)
                          for b, n in NATIVE.items()}
                    sar = {"VV": rng.normal(-12, 3, (120, 120)).astype(np.float32),
                           "VH": rng.normal(-19, 3, (120, 120)).astype(np.float32)}
                    sar["VH"][0, 0] = -np.inf
                    txn.put(pid.encode(), st_save(s2))
                    txn.put(s1.encode(), st_save(sar))
                    split = ("train", "validation", "test")[(h + v) % 3]
                    rows.append({"patch_id": pid, "labels": list(reben.CLASSES), "split": split,
                                 "country": "Austria", "s1_name": s1})
    env.close()
    meta = pd.DataFrame(rows)
    meta.to_parquet(root / reben.METADATA)
    return meta


@pytest.fixture(scope="module")
def fake_root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("reben")
    _fake_reben(root)
    return root


def test_records_follow_the_record_contract(fake_root: Path) -> None:
    recs = reben.load_records(fake_root)
    assert len(recs) == 3 * 36
    assert {r["split"] for r in recs} == {"train", "val", "test"}
    assert recs[0]["label"].shape == (19,) and recs[0]["label"].sum() == 19


def test_patch_load_resamples_to_120_and_keeps_dtypes(fake_root: Path) -> None:
    r = reben.load_records(fake_root)[0]
    opt, sar = r["optical"].load(), r["sar"].load()
    assert opt.shape == (12, 120, 120) and opt.dtype == np.uint16
    assert sar.shape == (2, 120, 120) and sar.dtype == np.float32
    assert np.isfinite(sar).all() and sar[1, 0, 0] == -50.0


def test_unknown_label_is_refused() -> None:
    with pytest.raises(ValueError, match="nomenclature"):
        reben.multi_hot(["Arable land", "Glaciers"])


def test_prepare_from_fake_reben(fake_root: Path, tmp_path: Path) -> None:
    import json

    from src.prepare_data import prepare, real_source

    recs = real_source(str(fake_root), tmp_path / "work")
    prepare(recs, tmp_path / "out", [1.0], temporal_steps=[1, 2], num_workers=2)
    tier = tmp_path / "out" / "full"
    manifest = json.loads((tier / "manifest.json").read_text())
    assert len(manifest["band_stats"]["optical"]["mean"]) == 12
    sample = StreamingDataset(str(tier / "test"))[0]
    assert sample["optical"].shape == (12, 120, 120) and "sar_shift_2" in sample
    assert sum(manifest["temporal"]["neighbour_found"]["1"].values()) == manifest["sizes"]["test"]
