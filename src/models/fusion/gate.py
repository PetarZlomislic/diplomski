import torch
from torch import Tensor, nn

from src.models.fusion.base import BaseFusion
from src.models.fusion.late import masked_softmax


class _Gate(BaseFusion):
    """Content-adaptive attention gate over per-modality feature maps (B, D, h, w).

    The gate network is the same two-layer MLP in both variants, applied either to pooled
    features (global) or per token as 1x1 convolutions (spatial), so the two arms have
    IDENTICAL parameter counts: a granularity effect cannot be a capacity effect.
    A dropped (None) modality is masked out of the softmax.
    `last_weights` holds the per-sample modality weights (B, M) of the latest forward,
    averaged over tokens for the spatial gate, for diagnostics.
    """

    spatial: bool

    def __init__(self, dim: int, modalities: tuple[str, ...], hidden: int = 64):
        super().__init__()
        self.modalities = tuple(modalities)
        self.out_dim = dim
        m = len(self.modalities)
        self.gate = nn.Sequential(nn.Conv2d(m * dim, hidden, 1), nn.ReLU(),
                                  nn.Conv2d(hidden, m, 1))
        self.last_weights: Tensor | None = None

    def forward(self, feats: dict[str, Tensor | None]) -> Tensor:
        ref = next((feats[m] for m in self.modalities if feats.get(m) is not None), None)
        if ref is None:
            raise ValueError("all modalities are None")
        maps = torch.stack([feats[m] if feats.get(m) is not None else torch.zeros_like(ref)
                            for m in self.modalities], 1)               # (B, M, D, h, w)
        present = torch.tensor([feats.get(m) is not None for m in self.modalities],
                               device=ref.device).view(1, -1, 1, 1)
        if not self.spatial:
            maps = maps.mean(dim=(3, 4), keepdim=True)                   # (B, M, D, 1, 1)
        scores = self.gate(maps.flatten(1, 2))                           # (B, M, h, w)
        w = masked_softmax(scores, present, dim=1)
        self.last_weights = w.detach().mean(dim=(2, 3))
        fused = (w.unsqueeze(2) * maps).sum(1)                           # (B, D, h, w)
        return fused.mean(dim=(2, 3))


class GlobalGateFusion(_Gate):
    """One weight per modality per sample: content-adaptive, spatially uniform."""

    spatial = False


class SpatialGateFusion(_Gate):
    """One weight per modality per token: content-adaptive, spatially varying."""

    spatial = True
