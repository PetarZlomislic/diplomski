"""Models, fusion mechanisms and the Lightning wrapper.

Model contract: an nn.Module whose forward takes a dict of named modality tensors (any of
which may be None: a dropped modality must not crash) and returns raw logits (B, num_classes),
never sigmoid. It sets `required_modalities`. Fusions declare a `level` that says what they
combine, so FusionNet knows where to call them:
    "input":   per-modality images  -> one image for a single shared encoder
    "feature": per-modality features -> one fused feature
    "logit":   per-modality logits  -> one set of logits
"""
from collections.abc import Callable
from typing import Any

import lightning as L
import torch
from hydra.utils import instantiate
from torch import Tensor, nn
from torchmetrics import MetricCollection
from torchmetrics.classification import MultilabelAccuracy, MultilabelF1Score

Feats = dict[str, Tensor | None]


def _first_present(feats: Feats, modalities) -> Tensor:
    ref = next((feats[m] for m in modalities if feats.get(m) is not None), None)
    if ref is None:
        raise ValueError("all modalities are None")
    return ref


# ---------------------------------------------------------------- encoder

class _Block(nn.Module):
    """ResNet basic block; `stride` 2 halves the resolution."""

    def __init__(self, cin: int, cout: int, stride: int):
        super().__init__()
        self.conv1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(cout)
        self.conv2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(cout)
        self.skip = (nn.Identity() if stride == 1 and cin == cout else
                     nn.Sequential(nn.Conv2d(cin, cout, 1, stride, bias=False),
                                   nn.BatchNorm2d(cout)))

    def forward(self, x: Tensor) -> Tensor:
        y = torch.relu(self.bn1(self.conv1(x)))
        return torch.relu(self.bn2(self.conv2(y)) + self.skip(x))


class ConvEncoder(nn.Module):
    """Small ResNet-style encoder: image -> feature map (B, out_dim, h, w).

    Shared initialisation across experiment arms (pre-analysis §6): the encoder is always
    built over the FULL band layout `bands` (e.g. {"optical": 12, "sar": 2}), initialised or
    loaded from `pretrained`, and only then is the stem sliced to `modalities`. So the
    optical encoder of `opt`, `late` and `gate_*` and the joint encoder of `early` all start
    from the same weights for a given seed; only the fusion layers and heads differ.

    Random init draws from a private generator seeded with `init_seed` (default: the global
    torch seed, i.e. the training seed), so it does not depend on what else the model
    builds or in which order.
    """

    def __init__(self, bands: dict[str, int], modalities: tuple[str, ...] | list[str],
                 width: int = 32, depth: int = 4, pretrained: str | None = None,
                 init_seed: int | None = None):
        super().__init__()
        self.modalities = tuple(modalities)
        unknown = set(self.modalities) - set(bands)
        if unknown:
            raise ValueError(f"modalities {sorted(unknown)} not in bands {list(bands)}")
        self.stem = nn.Sequential(nn.Conv2d(sum(bands.values()), width, 3, 1, 1, bias=False),
                                  nn.BatchNorm2d(width), nn.ReLU())
        dims = [width * 2**i for i in range(depth)]
        self.blocks = nn.Sequential(*[_Block(cin, cout, 2)
                                      for cin, cout in zip([width] + dims[:-1], dims)])
        self.out_dim = dims[-1]

        if pretrained:
            state = torch.load(pretrained, map_location="cpu", weights_only=True)
            self.load_state_dict(state.get("state_dict", state))
        else:
            g = torch.Generator().manual_seed(
                (torch.initial_seed() if init_seed is None else init_seed) % 2**63)
            for mod in self.modules():
                if isinstance(mod, nn.Conv2d):
                    nn.init.kaiming_normal_(mod.weight, mode="fan_out", nonlinearity="relu",
                                            generator=g)
                elif isinstance(mod, nn.BatchNorm2d):
                    nn.init.ones_(mod.weight)
                    nn.init.zeros_(mod.bias)

        # Slice the full-band stem to this encoder's bands.
        offsets, start = {}, 0
        for m, c in bands.items():
            offsets[m], start = (start, start + c), start + c
        idx = torch.cat([torch.arange(*offsets[m]) for m in self.modalities])
        sliced = nn.Conv2d(len(idx), width, 3, 1, 1, bias=False)
        with torch.no_grad():
            sliced.weight.copy_(self.stem[0].weight[:, idx])
        self.stem[0] = sliced
        self.in_channels = len(idx)

    def forward(self, x: Tensor) -> Tensor:
        return self.blocks(self.stem(x))


# ---------------------------------------------------------------- fusion

def masked_softmax(scores: Tensor, present: Tensor, dim: int) -> Tensor:
    """Softmax over `dim` with absent modalities given zero weight (the rest renormalised)."""
    return torch.softmax(scores.masked_fill(~present, float("-inf")), dim=dim)


class EarlyFusion(nn.Module):
    """Channel concatenation before the encoder. A dropped modality is zero-filled, which is
    the training mean after standardisation: early fusion has no way to isolate it.
    With a single modality this is the unimodal baseline (`opt`, `sar`)."""

    level = "input"

    def __init__(self, in_channels: dict[str, int], modalities: tuple[str, ...]):
        super().__init__()
        self.channels = {m: in_channels[m] for m in modalities}
        self.out_dim = sum(self.channels.values())

    def forward(self, feats: Feats) -> Tensor:
        ref = _first_present(feats, self.channels)
        b, _, h, w = ref.shape
        return torch.cat([feats[m] if feats.get(m) is not None
                          else ref.new_zeros(b, c, h, w) for m, c in self.channels.items()], 1)


class LateFusion(nn.Module):
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

    def forward(self, feats: Feats) -> Tensor:
        ref = _first_present(feats, self.modalities)
        present = [feats.get(m) is not None for m in self.modalities]
        w = masked_softmax(self.scores, torch.tensor(present, device=ref.device), dim=0)
        self.last_weights = w.detach().expand(ref.shape[0], -1)
        return sum(w[i] * feats[m] for i, m in enumerate(self.modalities) if present[i])


class _Gate(nn.Module):
    """Content-adaptive attention gate over per-modality feature maps (B, D, h, w).

    The gate network is the same two-layer MLP in both variants, applied either to pooled
    features (global) or per token as 1x1 convolutions (spatial), so the two arms have
    IDENTICAL parameter counts: a granularity effect cannot be a capacity effect.
    A dropped (None) modality is masked out of the softmax.
    `last_weights` holds the per-sample modality weights (B, M) of the latest forward,
    averaged over tokens for the spatial gate, for diagnostics.
    """

    level = "feature"
    spatial: bool

    def __init__(self, dim: int, modalities: tuple[str, ...], hidden: int = 64):
        super().__init__()
        self.modalities = tuple(modalities)
        self.out_dim = dim
        m = len(self.modalities)
        self.gate = nn.Sequential(nn.Conv2d(m * dim, hidden, 1), nn.ReLU(),
                                  nn.Conv2d(hidden, m, 1))
        self.last_weights: Tensor | None = None

    def forward(self, feats: Feats) -> Tensor:
        ref = _first_present(feats, self.modalities)
        maps = torch.stack([feats[m] if feats.get(m) is not None else torch.zeros_like(ref)
                            for m in self.modalities], 1)               # (B, M, D, h, w)
        present = torch.tensor([feats.get(m) is not None for m in self.modalities],
                               device=ref.device).view(1, -1, 1, 1)
        if not self.spatial:
            maps = maps.mean(dim=(3, 4), keepdim=True)                   # (B, M, D, 1, 1)
        w = masked_softmax(self.gate(maps.flatten(1, 2)), present, dim=1)  # (B, M, h, w)
        self.last_weights = w.detach().mean(dim=(2, 3))
        return (w.unsqueeze(2) * maps).sum(1).mean(dim=(2, 3))


class GlobalGateFusion(_Gate):
    """One weight per modality per sample: content-adaptive, spatially uniform."""

    spatial = False


class SpatialGateFusion(_Gate):
    """One weight per modality per token: content-adaptive, spatially varying."""

    spatial = True


# ---------------------------------------------------------------- models

class FusionNet(nn.Module):
    """The Experiment 1 model family: identical encoders, the fusion mechanism varies.

    `encoder` and `fusion` are partials (Hydra `_partial_: true`); FusionNet supplies the
    arguments that depend on the arm:
        fusion.level == "input"   one encoder over the fused input   (opt, sar, early)
        fusion.level == "feature" one encoder per modality -> fusion -> head   (gate_*)
        fusion.level == "logit"   one encoder + head per modality -> fusion    (late)
    """

    def __init__(self, num_classes: int, in_channels: dict[str, int], image_size: int,
                 modalities: list[str], encoder: Callable[..., ConvEncoder],
                 fusion: Callable[..., nn.Module]):
        super().__init__()
        self.required_modalities = mods = tuple(modalities)
        self.num_classes = num_classes
        self._in_channels = bands = dict(in_channels)
        self.level = level = getattr(fusion, "func", fusion).level
        if level == "input":
            self.fusion = fusion(in_channels=bands, modalities=mods)
            self.encoder = encoder(bands=bands, modalities=mods)
            self.head = nn.Linear(self.encoder.out_dim, num_classes)
            return
        if level not in ("feature", "logit"):
            raise ValueError(f"unknown fusion level {level!r}")
        self.encoders = nn.ModuleDict({m: encoder(bands=bands, modalities=(m,)) for m in mods})
        if level == "feature":
            self.fusion = fusion(dim=self.encoders[mods[0]].out_dim, modalities=mods)
            self.head = nn.Linear(self.fusion.out_dim, num_classes)
        else:
            self.heads = nn.ModuleDict(
                {m: nn.Linear(e.out_dim, num_classes) for m, e in self.encoders.items()})
            self.fusion = fusion(num_classes=num_classes, modalities=mods)

    @property
    def gate_weights(self) -> Tensor | None:
        """(B, M) modality weights of the latest forward, ordered as `required_modalities`;
        None for mechanisms without per-modality weights."""
        return getattr(self.fusion, "last_weights", None)

    def forward(self, batch: Feats) -> Tensor:
        x = {m: batch.get(m) for m in self.required_modalities}
        if all(v is None for v in x.values()):
            # Every input this arm uses is gone (e.g. `opt` under drop-optical): it sees the
            # standardised mean image, i.e. zeros, and predicts from that.
            ref = next(v for k, v in batch.items()
                       if k != "label" and isinstance(v, Tensor) and v.dim() == 4)
            b, _, h, w = ref.shape
            x = {m: ref.new_zeros(b, self._in_channels[m], h, w) for m in x}
        if self.level == "input":
            return self.head(self.encoder(self.fusion(x)).mean(dim=(2, 3)))
        feats = {m: self.encoders[m](v) if v is not None else None for m, v in x.items()}
        if self.level == "feature":
            return self.head(self.fusion(feats))
        return self.fusion({m: self.heads[m](f.mean(dim=(2, 3))) if f is not None else None
                            for m, f in feats.items()})


class DummyModel(nn.Module):
    """Per-modality spatial mean -> concatenation (a dropped modality is zero-filled, so the
    width is fixed) -> head. Ignores image content beyond the mean."""

    def __init__(self, num_classes: int, in_channels: dict[str, int], image_size: int,
                 hidden_dim: int = 0):
        super().__init__()
        self.in_channels = dict(in_channels)
        self.required_modalities = tuple(self.in_channels)
        dim = sum(self.in_channels.values())
        self.head = (nn.Sequential(nn.Linear(dim, hidden_dim), nn.ReLU(),
                                   nn.Linear(hidden_dim, num_classes))
                     if hidden_dim > 0 else nn.Linear(dim, num_classes))

    def forward(self, batch: Feats) -> Tensor:
        ref = _first_present(batch, self.required_modalities)
        feats = [batch[m].float().mean(dim=(2, 3)) if batch.get(m) is not None
                 else ref.new_zeros(ref.shape[0], c) for m, c in self.in_channels.items()]
        return self.head(torch.cat(feats, dim=1))


# ---------------------------------------------------------------- Lightning wrapper

def make_metrics(num_classes: int, threshold: float, per_class: bool = False) -> MetricCollection:
    """Scalar metrics; `per_class` adds "f1_per_class", a (num_classes,) vector, which
    cannot go through `log_dict`, so training never asks for it."""
    f1 = lambda avg: MultilabelF1Score(num_classes, threshold=threshold, average=avg)  # noqa: E731
    metrics = {"f1_macro": f1("macro"), "f1_micro": f1("micro"),
               "accuracy": MultilabelAccuracy(num_classes, threshold=threshold)}
    if per_class:
        metrics["f1_per_class"] = f1(None)
    return MetricCollection(metrics)


class LitModule(L.LightningModule):
    """Thin wrapper: owns loss, metrics and optimizer; delegates forward to a model.

    Multi-label task: BCEWithLogitsLoss on raw logits. The model is built from `model_cfg`
    in __init__ so a checkpoint's hparams are enough to reconstruct the architecture.
    """

    def __init__(self, model_cfg: dict[str, Any], optim_cfg: dict[str, Any], threshold: float,
                 sched_cfg: dict[str, Any] | None = None):
        super().__init__()
        self.save_hyperparameters()
        self.model = instantiate(model_cfg)
        self.loss_fn = nn.BCEWithLogitsLoss()
        num_classes = model_cfg["num_classes"]
        self.val_metrics = make_metrics(num_classes, threshold).clone(prefix="val/")
        self.test_metrics = make_metrics(num_classes, threshold).clone(prefix="test/")

    def forward(self, batch: Feats) -> Tensor:
        return self.model(batch)

    def _step(self, batch: dict[str, Tensor], stage: str) -> Tensor:
        logits = self(batch)
        loss = self.loss_fn(logits, batch["label"])
        self.log(f"{stage}/loss", loss, batch_size=logits.shape[0])
        if stage != "train":
            getattr(self, f"{stage}_metrics").update(logits, batch["label"].int())
        return loss

    def training_step(self, batch, batch_idx):
        return self._step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._step(batch, "val")

    def test_step(self, batch, batch_idx):
        return self._step(batch, "test")

    def _log_epoch(self, metrics: MetricCollection) -> None:
        self.log_dict(metrics.compute())
        metrics.reset()

    def on_validation_epoch_end(self) -> None:
        self._log_epoch(self.val_metrics)

    def on_test_epoch_end(self) -> None:
        self._log_epoch(self.test_metrics)

    def configure_optimizers(self) -> Any:
        optimizer = instantiate(self.hparams.optim_cfg, params=self.parameters())
        if not self.hparams.get("sched_cfg"):
            return optimizer
        scheduler = instantiate(self.hparams.sched_cfg, optimizer=optimizer)
        return {"optimizer": optimizer,
                "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"}}
