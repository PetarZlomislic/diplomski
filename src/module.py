from typing import Any

import lightning as L
import torch
from hydra.utils import instantiate
from torch import Tensor, nn


class LitModule(L.LightningModule):
    """Thin wrapper: owns loss and optimizer, delegates forward to a BaseModel.

    The model is built from `model_cfg` inside __init__ so the checkpoint's hparams
    are enough to reconstruct the architecture.
    """

    def __init__(self, model_cfg: dict[str, Any], optim_cfg: dict[str, Any]):
        super().__init__()
        self.save_hyperparameters()
        self.model = instantiate(model_cfg)
        self.loss_fn = nn.BCEWithLogitsLoss()

    def forward(self, batch: dict[str, Tensor | None]) -> Tensor:
        return self.model(batch)

    def _step(self, batch: dict[str, Tensor], stage: str) -> Tensor:
        logits = self(batch)
        loss = self.loss_fn(logits, batch["label"])
        self.log(f"{stage}/loss", loss, prog_bar=False, batch_size=logits.shape[0])
        return loss

    def training_step(self, batch: dict[str, Tensor], batch_idx: int) -> Tensor:
        return self._step(batch, "train")

    def validation_step(self, batch: dict[str, Tensor], batch_idx: int) -> Tensor:
        return self._step(batch, "val")

    def test_step(self, batch: dict[str, Tensor], batch_idx: int) -> Tensor:
        return self._step(batch, "test")

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return instantiate(self.hparams.optim_cfg, params=self.parameters())
