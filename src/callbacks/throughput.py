import time
from typing import Any

import lightning as L

from src.obs import RunLog


class ThroughputCallback(L.Callback):
    """samples/s and data_wait_frac: the fraction of wall time spent waiting on the
    dataloader (end of one train step -> start of the next). High values mean the job is
    input-bound. Accelerator work is async, so per-step compute time is approximate."""

    def __init__(self, log: RunLog, warn_data_wait_frac: float = 0.3):
        self.run_log = log
        self.warn_frac = warn_data_wait_frac
        self._reset()
        self._warned_epoch = -1

    WARMUP_BATCHES = 5

    def _reset(self) -> None:
        self.batches = 0
        self.samples = 0
        self.wait_s = 0.0
        self.compute_s = 0.0
        self._t_batch_end: float | None = None
        self._t_batch_start: float | None = None
        self._t_epoch_start = time.perf_counter()

    def on_train_epoch_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        self._reset()

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx) -> None:
        now = time.perf_counter()
        prev = self._t_batch_end if self._t_batch_end is not None else self._t_epoch_start
        self.wait_s += now - prev
        self._t_batch_start = now

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx) -> None:
        now = time.perf_counter()
        if self._t_batch_start is not None:
            self.compute_s += now - self._t_batch_start
        self._t_batch_end = now
        self.samples += len(batch["label"])
        self.batches += 1
        frac = self.data_wait_frac
        warmed_up = self.batches >= self.WARMUP_BATCHES  # first fetches include worker startup
        if warmed_up and frac > self.warn_frac and self._warned_epoch != trainer.current_epoch:
            self._warned_epoch = trainer.current_epoch
            self.run_log.emit("data_wait_high", level="warning", data_wait_frac=frac,
                          threshold=self.warn_frac, epoch=trainer.current_epoch)

    def on_train_epoch_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        self.run_log.emit("throughput", phase="train", epoch=trainer.current_epoch, **self.fields())

    @property
    def data_wait_frac(self) -> float | None:
        total = self.wait_s + self.compute_s
        return round(self.wait_s / total, 4) if total > 0 else None

    @property
    def samples_per_s(self) -> float | None:
        total = self.wait_s + self.compute_s
        return round(self.samples / total, 2) if total > 0 else None

    def fields(self) -> dict[str, Any]:
        return {"samples_per_s": self.samples_per_s, "data_wait_frac": self.data_wait_frac}
