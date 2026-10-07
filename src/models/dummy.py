import torch
from torch import Tensor, nn

from src.models.base import BaseModel


class DummyModel(BaseModel):
    """Per-modality spatial mean -> concat -> Linear. Missing modalities are zero-filled."""

    def __init__(
        self,
        num_classes: int,
        in_channels: dict[str, int],
        image_size: int,
        hidden_dim: int = 0,
        fusion: nn.Module | None = None,
    ):
        super().__init__()
        self.required_modalities = tuple(in_channels)
        self.in_channels = dict(in_channels)
        self.fusion = fusion
        in_dim = sum(self.in_channels.values())
        if hidden_dim > 0:
            self.head = nn.Sequential(
                nn.Linear(in_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, num_classes)
            )
        else:
            self.head = nn.Linear(in_dim, num_classes)

    def forward(self, batch: dict[str, Tensor | None]) -> Tensor:
        present = [batch[m] for m in self.required_modalities if batch.get(m) is not None]
        if not present:
            raise ValueError("all required modalities are None")
        b, device = present[0].shape[0], present[0].device
        feats: dict[str, Tensor | None] = {}
        for m, c in self.in_channels.items():
            x = batch.get(m)
            feats[m] = x.float().mean(dim=(2, 3)) if x is not None else None
        if self.fusion is not None:
            fused = self.fusion(feats)
        else:
            fused = torch.cat(
                [
                    f if f is not None else torch.zeros(b, self.in_channels[m], device=device)
                    for m, f in feats.items()
                ],
                dim=1,
            )
        return self.head(fused)
