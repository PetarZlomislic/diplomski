from typing import Any

import lightning as L
import torch
from hydra.utils import instantiate
from torch import Tensor, nn
from torchmetrics import MetricCollection
from torchmetrics.classification import MultilabelAccuracy, MultilabelF1Score


def make_metrics(num_classes: int, threshold: float) -> MetricCollection:
    return MetricCollection(
        {
            "f1_macro": MultilabelF1Score(num_classes, threshold=threshold, average="macro"),
            "f1_micro": MultilabelF1Score(num_classes, threshold=threshold, average="micro"),
            "accuracy": MultilabelAccuracy(num_classes, threshold=threshold),
        }
    )


class LitModule(L.LightningModule):
    """Thin wrapper: owns loss, metrics and optimizer; delegates forward to a BaseModel.

    Multi-label task: BCEWithLogitsLoss on raw logits. The model is built from `model_cfg`
    in __init__ so a checkpoint's hparams are enough to reconstruct the architecture.
    """

    def __init__(self, model_cfg: dict[str, Any], optim_cfg: dict[str, Any], threshold: float):
        super().__init__()
        self.save_hyperparameters()
        self.model = instantiate(model_cfg)
        self.loss_fn = nn.BCEWithLogitsLoss()
        num_classes = model_cfg["num_classes"]
        self.val_metrics = make_metrics(num_classes, threshold).clone(prefix="val/")
        self.test_metrics = make_metrics(num_classes, threshold).clone(prefix="test/")

    def forward(self, batch: dict[str, Tensor | None]) -> Tensor:
        return self.model(batch)

    def _step(self, batch: dict[str, Tensor], stage: str) -> Tensor:
        logits = self(batch)
        loss = self.loss_fn(logits, batch["label"])
        self.log(f"{stage}/loss", loss, batch_size=logits.shape[0])
        if stage == "val":
            self.val_metrics.update(logits, batch["label"].int())
        elif stage == "test":
            self.test_metrics.update(logits, batch["label"].int())
        return loss

    def training_step(self, batch: dict[str, Tensor], batch_idx: int) -> Tensor:
        return self._step(batch, "train")

    def validation_step(self, batch: dict[str, Tensor], batch_idx: int) -> Tensor:
        return self._step(batch, "val")

    def test_step(self, batch: dict[str, Tensor], batch_idx: int) -> Tensor:
        return self._step(batch, "test")

    def on_validation_epoch_end(self) -> None:
        self.log_dict(self.val_metrics.compute())
        self.val_metrics.reset()

    def on_test_epoch_end(self) -> None:
        self.log_dict(self.test_metrics.compute())
        self.test_metrics.reset()

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return instantiate(self.hparams.optim_cfg, params=self.parameters())
