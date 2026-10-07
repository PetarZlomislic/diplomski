from abc import ABC, abstractmethod
from torch import Tensor, nn


class BaseModel(nn.Module, ABC):
    """Base model contract for multi-modal classification.

    Takes a dict of named modality tensors, returns logits (no sigmoid).
    Modalities may be None (dropped). Returns logits of shape (B, num_classes).
    """

    required_modalities: tuple[str, ...]

    @abstractmethod
    def forward(self, batch: dict[str, Tensor | None]) -> Tensor:
        """Forward pass.

        Args:
            batch: dict with modality names as keys, Tensor or None as values.

        Returns:
            Logits of shape (batch_size, num_classes).
        """
