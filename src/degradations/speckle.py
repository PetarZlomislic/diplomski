import numpy as np
import torch
from torch import Tensor

from src.degradations.base import BaseDegradation


def looks(severity: float, looks_min: float) -> float:
    """Severity -> equivalent number of looks L. Speckle intensity variance is 1/L, and it is
    made linear in severity: 1/L = severity / looks_min. With looks_min=1 the grid
    {.25, .5, .75, 1} is L = {4, 2, 1.33, 1}; severity 1 is single-look speckle."""
    return looks_min / severity


class Speckle(BaseDegradation):
    """Multiplicative gamma speckle on SAR intensity: I' = I * n, n ~ Gamma(L, 1/L) (mean 1).

    Batches are standardised per band, so the noise is applied in physical units:
      scale="db":     x_dB' = x_dB + 10 log10(n)   ->  x' = x + 10 log10(n) / std
      scale="linear": I' = I * n                   ->  x' = ((x std + mean) n - mean) / std
    `bind` takes mean/std from the datamodule's `band_stats` (identity when it has none).
    The noise is independent per pixel; real speckle is spatially correlated (§7.2), and it
    adds to whatever speckle the product already has.
    """

    def __init__(self, modality: str = "sar", scale: str = "db", looks_min: float = 1.0):
        if scale not in ("db", "linear"):
            raise ValueError(f"scale must be 'db' or 'linear', not {scale!r}")
        self.modality = modality
        self.scale = scale
        self.looks_min = looks_min
        self.mean: Tensor | None = None
        self.std: Tensor | None = None

    def bind(self, datamodule) -> None:
        stats = (getattr(datamodule, "band_stats", None) or {}).get(self.modality)
        if stats:
            self.mean = torch.tensor(stats["mean"], dtype=torch.float32).view(-1, 1, 1)
            self.std = torch.tensor(stats["std"], dtype=torch.float32).view(-1, 1, 1)

    def __call__(
        self, batch: dict[str, Tensor | None], severity: float, generator: torch.Generator
    ) -> dict[str, Tensor | None]:
        out = dict(batch)
        x = batch.get(self.modality)
        if severity == 0.0 or x is None:
            return out
        L = looks(severity, self.looks_min)
        # torch has no seeded gamma sampler: derive a numpy one from the torch generator.
        seed = int(torch.randint(0, 2**62, (1,), generator=generator))
        n = torch.from_numpy(np.random.default_rng(seed).gamma(L, 1.0 / L, tuple(x.shape)))
        n = n.to(device=x.device, dtype=x.dtype)
        c = x.shape[1]
        mean = (self.mean if self.mean is not None else torch.zeros(c, 1, 1)).to(x)
        std = (self.std if self.std is not None else torch.ones(c, 1, 1)).to(x)
        if self.scale == "db":
            out[self.modality] = x + 10 * torch.log10(n.clamp_min(1e-12)) / std
        else:
            out[self.modality] = ((x * std + mean) * n - mean) / std
        return out
