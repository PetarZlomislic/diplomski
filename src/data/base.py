from typing import Any

import lightning as L
import torch
from torch import Tensor


def stats_key(key: str, stats: dict[str, Any]) -> str | None:
    """`sar` and `sar_shift_4` both standardise with the `sar` statistics."""
    if key in stats:
        return key
    base = key.split("_shift_")[0]
    return base if "_shift_" in key and base in stats else None


class BaseDataModule(L.LightningDataModule):
    """Batches are dicts:
        {"optical": Tensor[B,C,H,W], "sar": Tensor[B,2,H,W], "label": Tensor[B,K]}
    Modality keys are dataset-specific; "label" is mandatory.

    `data_uri` may be a local path OR a remote URI (hf://..., s3://...).
    Implementations MUST stream or mount, never download a multi-GB archive in setup().

    num_classes / in_channels / image_size are CONFIG-VISIBLE: model configs interpolate
    them, so they must be resolvable without instantiating the datamodule.

    Loaders may yield RAW batches (native dtypes, physical units). `prepare_batch` turns one
    into model input: float, standardised with `band_stats`, and (train only, if `augment`)
    randomly flipped. Lightning applies it on the device after transfer, so the batched
    work runs on the GPU, not in the loader workers; evaluation code applies it itself,
    before any degradation.

    `band_stats` ({modality: {"mean": [...], "std": [...]}}, physical units) is what batches
    are standardised with, or None if they are not standardised. Degradations that act in
    physical units (e.g. speckle) read it through `BaseDegradation.bind`.
    Test batches may carry `sar_shift_<k>` keys for the temporal degradation; models ignore
    keys they do not use.
    """

    num_classes: int
    in_channels: dict[str, int]
    image_size: int
    modalities: tuple[str, ...]
    data_uri: str
    band_stats: dict[str, dict[str, list[float]]] | None = None
    augment: bool = False

    def _stats(self, device: torch.device) -> dict[str, tuple[Tensor, Tensor]]:
        cache = self.__dict__.setdefault("_stats_cache", {})
        if device not in cache:
            cache[device] = {
                m: (torch.tensor(s["mean"], device=device).view(1, -1, 1, 1),
                    torch.tensor(s["std"], device=device).view(1, -1, 1, 1))
                for m, s in (self.band_stats or {}).items()}
        return cache[device]

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
