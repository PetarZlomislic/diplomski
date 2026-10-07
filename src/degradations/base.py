from abc import ABC, abstractmethod

import torch
from torch import Tensor


class BaseDegradation(ABC):
    """Applied to a batch at EVALUATION time only. Never during training.

    `generator` is separate from the training seed, so a given severity produces identical
    corruption regardless of which model or training seed is being evaluated.
    severity=0.0 must be an exact identity.
    """

    @abstractmethod
    def __call__(
        self,
        batch: dict[str, Tensor | None],
        severity: float,
        generator: torch.Generator,
    ) -> dict[str, Tensor | None]: ...
