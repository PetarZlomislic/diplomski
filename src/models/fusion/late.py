import torch
from torch import Tensor, nn

from src.models.fusion.base import BaseFusion


def masked_softmax(scores: Tensor, present: Tensor, dim: int) -> Tensor:
    """Softmax over `dim` with absent modalities given zero weight (the rest renormalised)."""
    return torch.softmax(scores.masked_fill(~present, float("-inf")), dim=dim)


class LateFusion(BaseFusion):
    """Weighted sum of per-modality logits with ONE learned weight per modality, shared by
    every sample: it can ensemble, but cannot reweight by content. A dropped (None)
    modality gets zero weight."""

    level = "logit"

    def __init__(self, num_classes: int, modalities: tuple[str, ...]):
        super().__init__()
        self.modalities = tuple(modalities)
        self.out_dim = num_classes
        self.scores = nn.Parameter(torch.zeros(len(self.modalities)))
        self.last_weights: Tensor | None = None

    def forward(self, feats: dict[str, Tensor | None]) -> Tensor:
        present = [feats.get(m) is not None for m in self.modalities]
        if not any(present):
            raise ValueError("all modalities are None")
        ref = next(feats[m] for m, p in zip(self.modalities, present) if p)
        w = masked_softmax(self.scores, torch.tensor(present, device=ref.device), dim=0)
        self.last_weights = w.detach().expand(ref.shape[0], -1)
        return sum(w[i] * feats[m] for i, m in enumerate(self.modalities) if present[i])
