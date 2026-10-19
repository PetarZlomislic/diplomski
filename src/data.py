"""Datamodules. Batches are dicts:
    {"optical": Tensor[B,C,H,W], "sar": Tensor[B,2,H,W], "label": Tensor[B,K]}
Modality keys are dataset-specific; "label" is mandatory.

num_classes / in_channels / image_size are CONFIG-VISIBLE: model configs interpolate them,
so they must be resolvable without instantiating the datamodule.

Loaders may yield RAW batches (native dtypes, physical units). `prepare_batch` turns one into
model input: float, standardised with `band_stats`, and (train only, if `augment`) randomly
flipped. Lightning applies it on the device after transfer, so the batched work runs on the
GPU, not in the loader workers; evaluation code applies it itself, before any degradation.

`band_stats` ({modality: {"mean": [...], "std": [...]}}, physical units) is what batches are
standardised with, or None if they are not. Degradations that act in physical units (e.g.
speckle) read it through `Degradation.bind`. Test batches may carry `sar_shift_<k>` keys for
the temporal degradation; models ignore keys they do not use.
"""
import json
from pathlib import Path
from typing import Any

import lightning as L
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from src.env import detect


def stats_key(key: str, stats: dict[str, Any]) -> str | None:
    """`sar` and `sar_shift_4` both standardise with the `sar` statistics."""
    if key in stats:
        return key
    base = key.split("_shift_")[0]
    return base if "_shift_" in key and base in stats else None


class BaseDataModule(L.LightningDataModule):
    band_stats: dict[str, dict[str, list[float]]] | None = None

    def __init__(self, num_classes: int, in_channels: dict[str, int], image_size: int,
                 batch_size: int, num_workers: int = 0, seed: int = 0, data_uri: str = "",
                 augment: bool = False):
        super().__init__()
        self.num_classes, self.image_size = num_classes, image_size
        self.in_channels = dict(in_channels)
        self.modalities = tuple(self.in_channels)
        self.batch_size, self.num_workers, self.seed = batch_size, num_workers, seed
        self.data_uri, self.augment = data_uri, augment
        self._stats_cache: dict[torch.device, dict[str, tuple[Tensor, Tensor]]] = {}

    def _stats(self, device: torch.device) -> dict[str, tuple[Tensor, Tensor]]:
        if device not in self._stats_cache:
            self._stats_cache[device] = {
                m: (torch.tensor(s["mean"], device=device).view(1, -1, 1, 1),
                    torch.tensor(s["std"], device=device).view(1, -1, 1, 1))
                for m, s in (self.band_stats or {}).items()}
        return self._stats_cache[device]

    def prepare_batch(self, batch: dict[str, Any], train: bool = False) -> dict[str, Any]:
        out = {k: v.float() if isinstance(v, Tensor) else v for k, v in batch.items()}
        ref = next((v for v in out.values() if isinstance(v, Tensor)), None)
        if ref is None:
            return out
        stats = self._stats(ref.device)
        for k, v in out.items():
            m = stats_key(k, stats)
            if m is not None and isinstance(v, Tensor):
                mean, std = stats[m]
                out[k] = (v - mean) / std
        if train and self.augment:
            images = [k for k, v in out.items() if isinstance(v, Tensor) and v.dim() == 4]
            n = out[images[0]].shape[0] if images else 0
            # One draw per sample, shared by all of its images so modalities stay aligned.
            h = (torch.rand(n, device=ref.device) < 0.5).view(-1, 1, 1, 1)
            w = (torch.rand(n, device=ref.device) < 0.5).view(-1, 1, 1, 1)
            for k in images:
                x = torch.where(h, out[k].flip(-1), out[k])
                out[k] = torch.where(w, x.flip(-2), x)
        return out

    def on_after_batch_transfer(self, batch: dict[str, Any], dataloader_idx: int) -> dict:
        trainer = getattr(self, "trainer", None)
        return self.prepare_batch(batch, train=bool(trainer and trainer.training))

    def train_dataloader(self):
        return self._loader(self.train_ds, shuffle=True)

    def val_dataloader(self):
        return self._loader(self.val_ds, shuffle=False)

    def test_dataloader(self):
        return self._loader(self.test_ds, shuffle=False)


class _DictTensorDataset(Dataset):
    def __init__(self, tensors: dict[str, Tensor]):
        self.tensors = tensors
        self.n = next(iter(tensors.values())).shape[0]

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, i: int) -> dict[str, Tensor]:
        return {k: v[i] for k, v in self.tensors.items()}


class DummyDataModule(BaseDataModule):
    """Deterministic random tensors from `seed`. No file I/O; ignores `data_uri`.
    Already standard normal, so `band_stats` is None. The test split carries
    `sar_shift_<k>` (another sample's SAR) for each of `temporal_steps`."""

    def __init__(self, num_classes: int, in_channels: dict[str, int], image_size: int,
                 n_samples: int = 64, batch_size: int = 8, num_workers: int = 0, seed: int = 0,
                 data_uri: str = "", temporal_steps: tuple[int, ...] = (1, 2, 4, 8)):
        super().__init__(num_classes, in_channels, image_size, batch_size, num_workers, seed,
                         data_uri)
        self.n_samples, self.temporal_steps = n_samples, tuple(temporal_steps)

    def _make_split(self, n: int, offset: int, shifts: bool = False) -> _DictTensorDataset:
        g = torch.Generator().manual_seed(self.seed + offset)
        tensors = {m: torch.randn(n, c, self.image_size, self.image_size, generator=g)
                   for m, c in self.in_channels.items()}
        tensors["label"] = torch.randint(0, 2, (n, self.num_classes), generator=g).float()
        if shifts and "sar" in tensors:
            for k in self.temporal_steps:
                tensors[f"sar_shift_{k}"] = tensors["sar"].roll(k, dims=0)
        return _DictTensorDataset(tensors)

    def setup(self, stage: str | None = None) -> None:
        n_val = n_test = max(1, self.n_samples // 5)
        self.train_ds = self._make_split(max(1, self.n_samples - n_val - n_test), 0)
        self.val_ds = self._make_split(n_val, 1)
        self.test_ds = self._make_split(n_test, 2, shifts=True)

    def _loader(self, ds: Dataset, shuffle: bool) -> DataLoader:
        return DataLoader(ds, batch_size=self.batch_size, shuffle=shuffle,
                          num_workers=self.num_workers)


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


def _to_tensors(sample: dict[str, Any]) -> dict[str, Any]:
    return {k: torch.as_tensor(v) if not isinstance(v, str) else v for k, v in sample.items()}


def _collate(samples: list[dict[str, Any]]) -> dict[str, Tensor | list[str]]:
    out: dict[str, Tensor | list[str]] = {}
    for k in samples[0]:
        vals = [s[k] for s in samples]
        out[k] = vals if isinstance(vals[0], str) else torch.stack(vals)  # native dtype
    return out


class StreamingDataModule(BaseDataModule):
    """Streams litdata shards written by src.prepare_data from `<data_uri>/<split>`.

    A relative `data_uri` is resolved against the runtime's data root (see src/env.py), so
    the same override works locally, on a mounted dataset, or on the Hub; file:// maps to a
    plain local path.

    Workers only decode and stack RAW samples (S2 uint16, S1 float32 dB). With `normalize`,
    `prepare_batch` standardises every band with `band_stats` (default: the manifest's, i.e.
    the tier's train split), so 0 is the training mean; `augment` adds per-sample random
    flips to the TRAIN split only.

    Nothing is downloaded or copied up front: local shards are read in place, remote ones
    are fetched chunk by chunk into a bounded cache.
    """

    def __init__(self, data_uri: str, num_classes: int, in_channels: dict[str, int],
                 image_size: int, batch_size: int = 32, num_workers: int = 0,
                 cache_dir: str | None = None, max_cache_size: str = "20GB", seed: int = 0,
                 normalize: bool = True,
                 band_stats: dict[str, dict[str, list[float]]] | None = None,
                 augment: bool = False):
        if data_uri.startswith("file://"):
            data_uri = data_uri[len("file://"):]
            if len(data_uri) > 2 and data_uri[0] == "/" and data_uri[2] == ":":  # /C:/...
                data_uri = data_uri[1:]
        super().__init__(num_classes, in_channels, image_size, batch_size, num_workers, seed,
                         detect().resolve_data_uri(data_uri), augment)
        self.cache_dir, self.max_cache_size, self.normalize = cache_dir, max_cache_size, normalize
        if normalize:
            stats = band_stats or load_manifest(self.data_uri).get("band_stats")
            if not stats:
                raise ValueError(f"no band_stats for {self.data_uri}: re-run prepare_data or "
                                 "set data.band_stats")
            self.band_stats = {m: dict(v) for m, v in stats.items()}

    def _dataset(self, split: str, shuffle: bool):
        from litdata import StreamingDataset

        base = self.data_uri.rstrip("/")
        return StreamingDataset(
            input_dir=f"{base}/{split}" if "://" in base else str(Path(base) / split),
            cache_dir=self.cache_dir, max_cache_size=self.max_cache_size, shuffle=shuffle,
            seed=self.seed, transform=_to_tensors)

    def setup(self, stage: str | None = None) -> None:
        self.train_ds = self._dataset("train", shuffle=True)
        self.val_ds = self._dataset("val", shuffle=False)
        self.test_ds = self._dataset("test", shuffle=False)

    def _loader(self, ds, shuffle: bool):
        from litdata import StreamingDataLoader

        return StreamingDataLoader(ds, batch_size=self.batch_size, num_workers=self.num_workers,
                                   collate_fn=_collate)
