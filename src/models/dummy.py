from torch import Tensor, nn

from src.models.base import BaseModel
from src.models.fusion.base import BaseFusion


class DummyModel(BaseModel):
    """Per-modality spatial mean -> fusion -> head. Ignores image content beyond the mean."""

    def __init__(
        self,
        num_classes: int,
        in_channels: dict[str, int],
        image_size: int,
        fusion: BaseFusion,
        hidden_dim: int = 0,
    ):
        super().__init__()
        self.required_modalities = tuple(in_channels)
        self.fusion = fusion
        if hidden_dim > 0:
            self.head = nn.Sequential(
                nn.Linear(fusion.out_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, num_classes)
            )
        else:
            self.head = nn.Linear(fusion.out_dim, num_classes)

    def forward(self, batch: dict[str, Tensor | None]) -> Tensor:
        feats = {
            m: batch[m].float().mean(dim=(2, 3)) if batch.get(m) is not None else None
            for m in self.required_modalities
        }
        return self.head(self.fusion(feats))
