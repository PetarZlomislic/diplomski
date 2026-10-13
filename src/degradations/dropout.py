import torch
from torch import Tensor

from src.degradations.base import BaseDegradation


class ModalityDropout(BaseDegradation):
    """Total sensor failure: any severity > 0 removes `modality` (it becomes None).
    Binary, so evaluate it at severities [0, 1]."""

    def __init__(self, modality: str):
        self.modality = modality

    def __call__(
        self, batch: dict[str, Tensor | None], severity: float, generator: torch.Generator
    ) -> dict[str, Tensor | None]:
        out = dict(batch)
        if severity > 0.0 and self.modality in out:
            out[self.modality] = None
        return out
