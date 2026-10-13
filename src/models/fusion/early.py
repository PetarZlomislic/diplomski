import torch
from torch import Tensor

from src.models.fusion.base import BaseFusion


class EarlyFusion(BaseFusion):
    """Channel concatenation before the encoder. A dropped modality is zero-filled, which is
    the training mean after standardisation: early fusion has no way to isolate it.
    With a single modality this is the unimodal baseline (`opt`, `sar`)."""

    level = "input"

    def __init__(self, in_channels: dict[str, int], modalities: tuple[str, ...]):
        super().__init__()
        self.channels = {m: in_channels[m] for m in modalities}
        self.out_dim = sum(self.channels.values())

    def forward(self, feats: dict[str, Tensor | None]) -> Tensor:
        ref = next((feats[m] for m in self.channels if feats.get(m) is not None), None)
        if ref is None:
            raise ValueError("all modalities are None")
        b, _, h, w = ref.shape
        return torch.cat([feats[m] if feats.get(m) is not None
                          else ref.new_zeros(b, c, h, w) for m, c in self.channels.items()], 1)
