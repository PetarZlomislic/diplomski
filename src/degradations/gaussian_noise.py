import torch
from torch import Tensor

from src.degradations.base import BaseDegradation


class GaussianNoise(BaseDegradation):
    """Additive N(0, (severity * sigma_max)^2) noise on one named modality."""

    def __init__(self, modality: str, sigma_max: float = 1.0):
        self.modality = modality
        self.sigma_max = sigma_max

    def __call__(
        self, batch: dict[str, Tensor | None], severity: float, generator: torch.Generator
    ) -> dict[str, Tensor | None]:
        out = dict(batch)
        x = batch.get(self.modality)
        if severity == 0.0 or x is None:
            return out
        noise = torch.randn(x.shape, generator=generator, dtype=x.dtype, device="cpu")
        out[self.modality] = x + severity * self.sigma_max * noise.to(x.device)
        return out
