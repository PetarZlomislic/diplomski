from abc import ABC, abstractmethod

from torch import Tensor, nn


class BaseFusion(nn.Module, ABC):
    """`level` says what the fusion combines, so a model knows where to call it:
        "feature": per-modality features -> one fused feature (default)
        "input":   per-modality images  -> one image for a single shared encoder
        "logit":   per-modality logits  -> one set of logits
    """

    out_dim: int
    level: str = "feature"

    @abstractmethod
    def forward(self, feats: dict[str, Tensor | None]) -> Tensor:
        """Per-modality tensors -> one fused tensor.
        MUST handle a None value (dropped modality) without crashing."""
