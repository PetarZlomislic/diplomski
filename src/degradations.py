"""Evaluation-time degradations. Never applied during training.

A degradation maps (batch, severity, generator) -> batch. `generator` is separate from the
training seed, so a given severity produces identical corruption regardless of which model
or training seed is being evaluated. severity=0.0 is an exact identity.
"""
import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

Batch = dict[str, Tensor | None]


class Degradation:
    """The identity, and the base of every degradation that acts on ONE `modality`:
    subclasses implement `apply`, which only runs for severity > 0 on a present modality."""

    modality: str | None = None

    def bind(self, datamodule) -> None:
        """Called once with the evaluation datamodule before any batch, for degradations
        that need dataset facts (e.g. `band_stats` to work in physical units)."""

    def __call__(self, batch: Batch, severity: float, generator: torch.Generator) -> Batch:
        out = dict(batch)
        if severity != 0.0 and out.get(self.modality) is not None:
            out[self.modality] = self.apply(out[self.modality], severity, generator, batch)
        return out

    def apply(self, x: Tensor, severity: float, generator: torch.Generator,
              batch: Batch) -> Tensor | None:
        return x


class GaussianNoise(Degradation):
    """Additive N(0, (severity * sigma_max)^2) noise on one named modality."""

    def __init__(self, modality: str, sigma_max: float = 1.0):
        self.modality, self.sigma_max = modality, sigma_max

    def apply(self, x, severity, generator, batch):
        noise = torch.randn(x.shape, generator=generator, dtype=x.dtype, device="cpu")
        return x + severity * self.sigma_max * noise.to(x.device)


class ModalityDropout(Degradation):
    """Total sensor failure: any severity > 0 removes `modality` (it becomes None).
    Binary, so evaluate it at severities [0, 1]."""

    def __init__(self, modality: str):
        self.modality = modality

    def apply(self, x, severity, generator, batch):
        return None


def cloud_mask(n: int, h: int, w: int, fraction: float, generator: torch.Generator,
               octaves: tuple[int, ...] = (2, 4, 8), persistence: float = 0.5) -> Tensor:
    """Boolean (n, h, w) masks covering EXACTLY round(fraction * h * w) pixels each.

    A smooth random field (value noise: Gaussian grids of `octaves` cells per side,
    bilinearly upsampled and summed with amplitude persistence**i) is ranked per sample and
    its top pixels are masked. The mask is coherent blobs rather than salt and pepper, and
    the occluded fraction is exact rather than expected.
    """
    k = round(fraction * h * w)
    if k == 0:
        return torch.zeros(n, h, w, dtype=torch.bool)
    field = torch.zeros(n, 1, h, w)
    for i, cells in enumerate(octaves):
        grid = torch.randn(n, 1, cells + 1, cells + 1, generator=generator)
        field += persistence**i * F.interpolate(grid, size=(h, w), mode="bilinear",
                                                align_corners=True)
    flat = field.view(n, -1)
    idx = torch.topk(flat, k, dim=1, sorted=False).indices
    return torch.zeros_like(flat, dtype=torch.bool).scatter_(1, idx, True).view(n, h, w)


class CloudOcclusion(Degradation):
    """Synthetic cloud: severity = exact fraction of each patch occluded. Inside the mask
    every band of `modality` is replaced by `fill` (0.0 = the training-set band mean after
    standardisation): total information loss, not a bright cloud signature.
    This simulates localised occlusion, not realistic cloud (pre-analysis §7.1)."""

    def __init__(self, modality: str = "optical", fill: float = 0.0,
                 octaves: tuple[int, ...] = (2, 4, 8), persistence: float = 0.5):
        self.modality, self.fill = modality, fill
        self.octaves, self.persistence = tuple(octaves), persistence

    def apply(self, x, severity, generator, batch):
        n, _, h, w = x.shape
        mask = cloud_mask(n, h, w, severity, generator, self.octaves, self.persistence)
        return x.masked_fill(mask.to(x.device).unsqueeze(1), self.fill)


def looks(severity: float, looks_min: float) -> float:
    """Severity -> equivalent number of looks L. Speckle intensity variance is 1/L, and it is
    made linear in severity: 1/L = severity / looks_min. With looks_min=1 the grid
    {.25, .5, .75, 1} is L = {4, 2, 1.33, 1}; severity 1 is single-look speckle."""
    return looks_min / severity


class Speckle(Degradation):
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
        self.modality, self.scale, self.looks_min = modality, scale, looks_min
        self.mean: Tensor | None = None
        self.std: Tensor | None = None

    def bind(self, datamodule) -> None:
        stats = (getattr(datamodule, "band_stats", None) or {}).get(self.modality)
        if stats:
            self.mean = torch.tensor(stats["mean"], dtype=torch.float32).view(-1, 1, 1)
            self.std = torch.tensor(stats["std"], dtype=torch.float32).view(-1, 1, 1)

    def apply(self, x, severity, generator, batch):
        L = looks(severity, self.looks_min)
        # torch has no seeded gamma sampler: derive a numpy one from the torch generator.
        seed = int(torch.randint(0, 2**62, (1,), generator=generator))
        n = torch.from_numpy(np.random.default_rng(seed).gamma(L, 1.0 / L, tuple(x.shape)))
        n = n.to(device=x.device, dtype=x.dtype)
        c = x.shape[1]
        mean = (self.mean if self.mean is not None else torch.zeros(c, 1, 1)).to(x)
        std = (self.std if self.std is not None else torch.ones(c, 1, 1)).to(x)
        if self.scale == "db":
            return x + 10 * torch.log10(n.clamp_min(1e-12)) / std
        return ((x * std + mean) * n - mean) / std


class TemporalMisalignment(Degradation):
    """Replaces `modality` with a non-corresponding patch `severity` steps away.

    reBEN has one S1/S2 pair per patch, so a step is a SPATIAL offset (pre-analysis §7.6
    fallback): step k pairs each optical patch with the SAR patch k grid cells away in the
    same tile (k * 1.2 km at 120 px and 10 m). src.prepare_data precomputes the shifted SAR
    into the test shards as `<modality>_shift_<k>`, because neighbours must come from the
    full source, not the subsampled tier. Severity must be one of the prepared steps.
    """

    def __init__(self, modality: str = "sar"):
        self.modality = modality

    def apply(self, x, severity, generator, batch):
        step = int(severity)
        if step != severity:
            raise ValueError(f"temporal severity must be an integer step, got {severity}")
        key = f"{self.modality}_shift_{step}"
        if key not in batch:
            have = sorted(k for k in batch if k.startswith(f"{self.modality}_shift_"))
            raise KeyError(f"no {key} in the batch (prepared: {have}); re-run prepare_data "
                           f"with --temporal-steps including {step}")
        return batch[key]
