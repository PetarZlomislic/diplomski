import torch
from torch import Tensor

from src.degradations.base import BaseDegradation


class NoDegradation(BaseDegradation):
    def __call__(
        self, batch: dict[str, Tensor | None], severity: float, generator: torch.Generator
    ) -> dict[str, Tensor | None]:
        return dict(batch)
