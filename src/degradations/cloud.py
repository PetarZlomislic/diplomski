import torch
import torch.nn.functional as F
from torch import Tensor

from src.degradations.base import BaseDegradation


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
    mask = torch.zeros_like(flat, dtype=torch.bool).scatter_(1, idx, True)
    return mask.view(n, h, w)


class CloudOcclusion(BaseDegradation):
    """Synthetic cloud: severity = exact fraction of each patch occluded. Inside the mask
    every band of `modality` is replaced by `fill` (0.0 = the training-set band mean after
    standardisation): total information loss, not a bright cloud signature.
    This simulates localised occlusion, not realistic cloud (pre-analysis §7.1)."""

    def __init__(self, modality: str = "optical", fill: float = 0.0,
                 octaves: tuple[int, ...] = (2, 4, 8), persistence: float = 0.5):
        self.modality = modality
        self.fill = fill
        self.octaves = tuple(octaves)
        self.persistence = persistence

    def __call__(
        self, batch: dict[str, Tensor | None], severity: float, generator: torch.Generator
    ) -> dict[str, Tensor | None]:
        out = dict(batch)
        x = batch.get(self.modality)
        if severity == 0.0 or x is None:
            return out
        n, _, h, w = x.shape
        mask = cloud_mask(n, h, w, severity, generator, self.octaves, self.persistence)
        out[self.modality] = x.masked_fill(mask.to(x.device).unsqueeze(1), self.fill)
        return out
