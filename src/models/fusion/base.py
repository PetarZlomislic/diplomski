from abc import ABC, abstractmethod

from torch import Tensor, nn


class BaseFusion(nn.Module, ABC):
    out_dim: int

    @abstractmethod
    def forward(self, feats: dict[str, Tensor | None]) -> Tensor:
        """Per-modality features -> one fused feature tensor.
        MUST handle a None value (dropped modality) without crashing."""
