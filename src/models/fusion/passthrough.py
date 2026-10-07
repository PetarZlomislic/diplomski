import torch
from torch import Tensor

from src.models.fusion.base import BaseFusion


class PassthroughFusion(BaseFusion):
    """Single modality passes through; multiple concatenate. A None modality is zero-filled
    so the output width is fixed."""

    def __init__(self, in_dims: dict[str, int]):
        super().__init__()
        self.in_dims = dict(in_dims)
        self.out_dim = sum(self.in_dims.values())

    def forward(self, feats: dict[str, Tensor | None]) -> Tensor:
        present = [f for f in feats.values() if f is not None]
        if not present:
            raise ValueError("all modalities are None")
        ref = present[0]
        parts = [
            feats[m] if feats.get(m) is not None
            else torch.zeros(ref.shape[0], d, device=ref.device, dtype=ref.dtype)
            for m, d in self.in_dims.items()
        ]
        return parts[0] if len(parts) == 1 else torch.cat(parts, dim=1)
