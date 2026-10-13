import json
from pathlib import Path
from typing import Any

import torch
from litdata import StreamingDataLoader, StreamingDataset
from torch import Tensor

from src.data.base import BaseDataModule
from src.env import detect


def _normalize_uri(uri: str) -> str:
    """litdata reads local dirs and remote URIs (hf://, s3://, gs://) in place; file:// is
    mapped to a plain local path."""
    if uri.startswith("file://"):
        uri = uri[len("file://"):]
        if len(uri) > 2 and uri[0] == "/" and uri[2] == ":":  # file:///C:/... on Windows
            uri = uri[1:]
    return uri


def _to_tensors(sample: dict[str, Any]) -> dict[str, Any]:
    return {k: torch.as_tensor(v) if not isinstance(v, str) else v for k, v in sample.items()}


def load_manifest(data_uri: str) -> dict[str, Any]:
    """`<data_uri>/manifest.json`, written by src.prepare_data. Local paths and hf:// only."""
    base = data_uri.rstrip("/")
    if base.startswith("hf://"):
        from huggingface_hub import HfFileSystem

        with HfFileSystem().open(f"{base[len('hf://'):]}/manifest.json") as f:
            return json.load(f)
    if "://" in base:
        raise ValueError(f"cannot read the manifest from {base}; set data.band_stats instead")
    return json.loads((Path(base) / "manifest.json").read_text())


class StreamingDataModule(BaseDataModule):
    """Streams litdata shards written by src.prepare_data from `<data_uri>/<split>`.

    A relative `data_uri` is resolved against the runtime's data root (see src/env.py), so
    the same override works locally, on a mounted dataset, or on the Hub.

    Workers only decode and stack RAW samples (S2 uint16, S1 float32 dB). With `normalize`,
    `prepare_batch` (on the GPU, see BaseDataModule) standardises every band with
    `band_stats` (default: the manifest's, i.e. the tier's train split), so 0 is the
    training mean; `augment` adds per-sample random flips to the TRAIN split only.

    Nothing is downloaded or copied up front: local shards are read in place, remote ones
    are fetched chunk by chunk into a bounded cache.
    """

    def __init__(
        self,
        data_uri: str,
        num_classes: int,
        in_channels: dict[str, int],
        image_size: int,
        batch_size: int = 32,
        num_workers: int = 0,
        cache_dir: str | None = None,
        max_cache_size: str = "20GB",
        seed: int = 0,
        normalize: bool = True,
        band_stats: dict[str, dict[str, list[float]]] | None = None,
        augment: bool = False,
    ):
        super().__init__()
        self.data_uri = detect().resolve_data_uri(_normalize_uri(data_uri))
        self.num_classes = num_classes
        self.in_channels = dict(in_channels)
        self.image_size = image_size
        self.modalities = tuple(self.in_channels)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.cache_dir = cache_dir
        self.max_cache_size = max_cache_size
        self.seed = seed
        self.normalize = normalize
        self.augment = augment
        self.band_stats: dict[str, dict[str, list[float]]] | None = None
        if normalize:
            stats = band_stats or load_manifest(self.data_uri).get("band_stats")
            if not stats:
                raise ValueError(f"no band_stats for {self.data_uri}: re-run prepare_data or "
                                 "set data.band_stats")
            self.band_stats = {m: dict(v) for m, v in stats.items()}

    def _dataset(self, split: str, shuffle: bool) -> StreamingDataset:
        base = self.data_uri.rstrip("/")
        input_dir = f"{base}/{split}" if "://" in base else str(Path(base) / split)
        return StreamingDataset(
            input_dir=input_dir,
            cache_dir=self.cache_dir,
            max_cache_size=self.max_cache_size,
            shuffle=shuffle,
            seed=self.seed,
            transform=_to_tensors,
        )

    def setup(self, stage: str | None = None) -> None:
        self.train_ds = self._dataset("train", shuffle=True)
        self.val_ds = self._dataset("val", shuffle=False)
        self.test_ds = self._dataset("test", shuffle=False)

    def _loader(self, ds: StreamingDataset) -> StreamingDataLoader:
        return StreamingDataLoader(ds, batch_size=self.batch_size, num_workers=self.num_workers,
                                   collate_fn=_collate)

    def train_dataloader(self) -> StreamingDataLoader:
        return self._loader(self.train_ds)

    def val_dataloader(self) -> StreamingDataLoader:
        return self._loader(self.val_ds)

    def test_dataloader(self) -> StreamingDataLoader:
        return self._loader(self.test_ds)


def _collate(samples: list[dict[str, Any]]) -> dict[str, Tensor | list[str]]:
    out: dict[str, Tensor | list[str]] = {}
    for k in samples[0]:
        vals = [s[k] for s in samples]
        out[k] = vals if isinstance(vals[0], str) else torch.stack(vals)  # native dtype
    return out
