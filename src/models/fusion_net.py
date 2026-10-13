from collections.abc import Callable

from torch import Tensor, nn

from src.models.base import BaseModel
from src.models.encoders import ConvEncoder
from src.models.fusion.base import BaseFusion


class FusionNet(BaseModel):
    """The Experiment 1 model family: identical encoders, the fusion mechanism varies.

    `encoder` and `fusion` are partials (Hydra `_partial_: true`); FusionNet supplies the
    arguments that depend on the arm:
        fusion.level == "input"   one encoder over the fused input   (opt, sar, early)
        fusion.level == "feature" one encoder per modality -> fusion -> head   (gate_*)
        fusion.level == "logit"   one encoder + head per modality -> fusion    (late)
    """

    def __init__(
        self,
        num_classes: int,
        in_channels: dict[str, int],
        image_size: int,
        modalities: list[str],
        encoder: Callable[..., ConvEncoder],
        fusion: Callable[..., BaseFusion],
    ):
        super().__init__()
        self.required_modalities = tuple(modalities)
        self.num_classes = num_classes
        self._in_channels = dict(in_channels)
        mods = self.required_modalities
        level = getattr(fusion, "func", fusion).level
        if level == "input":
            self.fusion = fusion(in_channels=dict(in_channels), modalities=mods)
            self.encoder = encoder(bands=dict(in_channels), modalities=mods)
            self.head = nn.Linear(self.encoder.out_dim, num_classes)
        elif level == "feature":
            self.encoders = nn.ModuleDict(
                {m: encoder(bands=dict(in_channels), modalities=(m,)) for m in mods})
            dim = self.encoders[mods[0]].out_dim
            self.fusion = fusion(dim=dim, modalities=mods)
            self.head = nn.Linear(self.fusion.out_dim, num_classes)
        elif level == "logit":
            self.encoders = nn.ModuleDict(
                {m: encoder(bands=dict(in_channels), modalities=(m,)) for m in mods})
            self.heads = nn.ModuleDict(
                {m: nn.Linear(e.out_dim, num_classes) for m, e in self.encoders.items()})
            self.fusion = fusion(num_classes=num_classes, modalities=mods)
        else:
            raise ValueError(f"unknown fusion level {level!r}")
        self.level = level

    @property
    def gate_weights(self) -> Tensor | None:
        """(B, M) modality weights of the latest forward, ordered as `required_modalities`;
        None for mechanisms without per-modality weights."""
        return getattr(self.fusion, "last_weights", None)

    def forward(self, batch: dict[str, Tensor | None]) -> Tensor:
        x = {m: batch.get(m) for m in self.required_modalities}
        if all(v is None for v in x.values()):
            # Every input this arm uses is gone (e.g. `opt` under drop-optical): it sees the
            # standardised mean image, i.e. zeros, and predicts from that.
            ref = next(v for k, v in batch.items()
                       if k != "label" and isinstance(v, Tensor) and v.dim() == 4)
            b, _, h, w = ref.shape
            channels = {m: c for m, c in self._in_channels.items() if m in x}
            x = {m: ref.new_zeros(b, c, h, w) for m, c in channels.items()}
        if self.level == "input":
            return self.head(self.encoder(self.fusion(x)).mean(dim=(2, 3)))
        feats = {m: self.encoders[m](v) if v is not None else None for m, v in x.items()}
        if self.level == "feature":
            return self.head(self.fusion(feats))
        logits = {m: self.heads[m](f.mean(dim=(2, 3))) if f is not None else None
                  for m, f in feats.items()}
        return self.fusion(logits)
