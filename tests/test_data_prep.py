from pathlib import Path

import pytest
from litdata import StreamingDataset

from src.prepare_data import (
    BANDS,
    SPLITS,
    TIERS,
    ClassFloorError,
    prepare,
    subset,
    synthetic_source,
)
from src.train import run
from tests.conftest import compose_cfg


@pytest.fixture(scope="module")
def records() -> list[dict]:
    return synthetic_source(n=1500, seed=0)


@pytest.fixture(scope="module")
def shards(records, tmp_path_factory: pytest.TempPathFactory) -> Path:
    out = tmp_path_factory.mktemp("prepared")
    prepare(records, out, TIERS.values(), min_per_class=1)
    return out


def test_sampling_is_within_split(records) -> None:
    out = subset(records, 0.05, min_per_class=1)
    source_split = {r["patch_id"]: r["split"] for r in records}
    for split, recs in out.items():
        assert recs, split
        assert all(source_split[r["patch_id"]] == split for r in recs)
    source_test = {r["patch_id"] for r in records if r["split"] == "test"}
    assert not source_test & {r["patch_id"] for r in out["train"]}


def test_class_floor_holds_on_normal_distribution(records) -> None:
    out = subset(records, 0.01, min_per_class=2)
    for split in SPLITS:
        counts = sum(r["label"] for r in out[split])
        assert counts.min() >= 2, (split, counts)


def test_class_floor_assertion_fires_on_skewed_distribution() -> None:
    skewed = synthetic_source(n=1500, seed=0, starve_class=18)
    with pytest.raises(ClassFloorError) as exc:
        subset(skewed, 0.05, min_per_class=1)
    msg = str(exc.value)
    assert "'val': {18: 0}" in msg and "'test': {18: 0}" in msg
    assert "'train'" not in msg


@pytest.mark.parametrize("tier", list(TIERS))
def test_all_14_bands_survive_every_tier(shards: Path, tier: str) -> None:
    for split in SPLITS:
        ds = StreamingDataset(str(shards / tier / split))
        sample = ds[0]
        assert {m: sample[m].shape[0] for m in BANDS} == {"optical": 12, "sar": 2}
        assert sum(sample[m].shape[0] for m in BANDS) == 14


def _train_streaming(uri: str, out: Path, cache: Path) -> Path:
    cfg = compose_cfg(
        "data=streaming", f"data.data_uri={uri}", "data.image_size=8", "data.batch_size=8",
        f"data.cache_dir={cache.as_posix()}", "trainer.max_epochs=1", "trainer.accelerator=cpu",
        f"output_dir={out.as_posix()}",
    )
    return run(cfg, {"model": "dummy", "data": "streaming"})


@pytest.mark.parametrize("scheme", ["path", "file_uri"])
def test_streaming_trains_one_epoch_without_copying(shards: Path, tmp_path: Path,
                                                    scheme: str) -> None:
    dev = (shards / "dev").resolve()
    uri = dev.as_posix() if scheme == "path" else dev.as_uri()
    cache = tmp_path / "cache"
    run_dir = _train_streaming(uri, tmp_path / "runs", cache)
    assert (run_dir / "checkpoints" / "last.ckpt").exists()
    copied = list(cache.rglob("*.bin")) if cache.exists() else []
    assert not copied, f"shards were copied into the cache: {copied[:3]}"


def test_streaming_relative_uri_uses_data_root(shards: Path, tmp_path: Path,
                                               monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DATA_ROOT", str(shards))
    run_dir = _train_streaming("dev", tmp_path / "runs", tmp_path / "cache")
    assert (run_dir / "checkpoints" / "last.ckpt").exists()
