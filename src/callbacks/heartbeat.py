import threading
import time
from collections.abc import Callable
from typing import Any

import lightning as L

from src.obs import RunLog


class Heartbeat:
    """Emits a `heartbeat` line every `interval_s` of WALL time from a background thread, so
    a hung step still produces lines (with a frozen step) instead of going silent."""

    def __init__(self, log: RunLog, interval_s: float,
                 sources: list[Callable[[], dict[str, Any]]] | None = None):
        self.log = log
        self.interval_s = interval_s
        self.sources = sources or []
        self.state: dict[str, Any] = {}
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._t0 = time.monotonic()

    def update(self, **state: Any) -> None:
        self.state.update(state)

    def beat(self) -> None:
        fields = dict(self.state)
        for src in self.sources:
            try:
                fields.update(src())
            except Exception:
                pass
        step, max_steps = fields.get("step"), fields.get("max_steps")
        elapsed = time.monotonic() - self._t0
        if step and max_steps and max_steps > step:
            fields["eta_s"] = round(elapsed / step * (max_steps - step))
        fields["elapsed_s"] = round(elapsed, 1)
        self.log.emit("heartbeat", **fields)

    def _loop(self) -> None:
        self.beat()
        while not self._stop.wait(self.interval_s):
            self.beat()

    def start(self) -> None:
        if self._thread is None:
            self._t0 = time.monotonic()
            self._stop.clear()
            self._thread = threading.Thread(target=self._loop, name="heartbeat", daemon=True)
            self._thread.start()

    def stop(self) -> None:
        if self._thread is not None:
            self._stop.set()
            self._thread.join(timeout=5)
            self._thread = None


class HeartbeatCallback(L.Callback):
    """Feeds training state into a Heartbeat and runs it for the duration of fit()."""

    def __init__(self, heartbeat: Heartbeat):
        self.hb = heartbeat

    def on_fit_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        self.hb.update(phase="train", epoch=trainer.current_epoch, step=trainer.global_step,
                       max_steps=trainer.estimated_stepping_batches)
        self.hb.start()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx) -> None:
        loss = outputs["loss"] if isinstance(outputs, dict) else outputs
        lr = trainer.optimizers[0].param_groups[0]["lr"] if trainer.optimizers else None
        self.hb.update(phase="train", epoch=trainer.current_epoch, step=trainer.global_step,
                       loss=float(loss) if loss is not None else None, lr=lr)

    def on_validation_start(self, trainer, pl_module) -> None:
        self.hb.update(phase="val")

    def on_validation_end(self, trainer, pl_module) -> None:
        self.hb.update(phase="train")

    def on_fit_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        self.hb.stop()

    def on_exception(self, trainer, pl_module, exception: BaseException) -> None:
        self.hb.stop()
