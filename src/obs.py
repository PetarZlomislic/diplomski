"""Observability: structured JSONL to stdout + <run dir>/run.jsonl, and run provenance.

Every line: {"ts", "level", "event", "run_id", ...event fields}. Remote jobs are detached,
so this log is the only window into a run.
"""
import functools
import hashlib
import json
import logging
import math
import os
import platform
import secrets
import subprocess
import sys
import threading
import time
import traceback
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TextIO

import lightning as L
import psutil

REPO_ROOT = Path(__file__).resolve().parent.parent


def new_run_id() -> str:
    """Timestamp + 6 hex chars, generated once per run."""
    return datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S") + "-" + secrets.token_hex(3)


def _git(*args: str) -> str | None:
    try:
        out = subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True, text=True,
                             check=True)
        return out.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def git_sha() -> str:
    """HEAD of the checkout, else REPO_SHA from a remote bootstrap (code fetched as a
    tarball has no .git), else "unknown"."""
    return _git("rev-parse", "HEAD") or os.environ.get("REPO_SHA") or "unknown"


def git_dirty() -> bool | None:
    status = _git("status", "--porcelain", "--untracked-files=no")
    if status is None:
        return False if os.environ.get("REPO_SHA") else None  # pinned tarball: clean
    return bool(status)


def config_hash(resolved_cfg: dict[str, Any]) -> str:
    blob = json.dumps(resolved_cfg, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:12]


def _clean(v: Any) -> Any:
    """JSON-safe: non-finite floats -> None, tensors/paths -> python/str."""
    if hasattr(v, "item") and callable(v.item) and getattr(v, "numel", lambda: 1)() == 1:
        v = v.item()
    if isinstance(v, float) and not math.isfinite(v):
        return None
    if isinstance(v, dict):
        return {str(k): _clean(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_clean(x) for x in v]
    if isinstance(v, (str, int, float, bool)) or v is None:
        return v
    return str(v)


class RunLog:
    """Thread-safe JSONL emitter with a dual sink: stdout always, plus a file."""

    def __init__(self, run_id: str, path: Path | None, stream: TextIO | None = None):
        self.run_id = run_id
        self.path = path
        self._stream = stream or sys.stdout
        self._lock = threading.Lock()
        self._file = None
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            self._file = path.open("a", encoding="utf-8")

    def emit(self, event: str, level: str = "info", **fields: Any) -> None:
        record = {
            "ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            "level": level,
            "event": event,
            "run_id": self.run_id,
            **_clean(fields),
        }
        line = json.dumps(record, allow_nan=False)
        with self._lock:
            self._stream.write(line + "\n")
            self._stream.flush()
            if self._file is not None:
                self._file.write(line + "\n")
                self._file.flush()

    def close(self) -> None:
        with self._lock:
            if self._file is not None:
                self._file.close()
                self._file = None


class _JsonlHandler(logging.Handler):
    """Routes stdlib logging (ours and libraries') into the JSONL stream."""

    def __init__(self, log: RunLog):
        super().__init__(level=logging.WARNING)
        self.log = log

    def emit(self, record: logging.LogRecord) -> None:
        self.log.emit("log", level=record.levelname.lower(), logger=record.name,
                      message=record.getMessage())


def setup_logging(run_id: str, output_dir: Path, to_file: bool = True) -> RunLog:
    """Create the run's JSONL log (stdout, plus <output_dir>/run.jsonl unless `to_file` is
    False) and route warnings from stdlib logging into it."""
    log = RunLog(run_id, output_dir / "run.jsonl" if to_file else None)
    root = logging.getLogger()
    for h in [h for h in root.handlers if isinstance(h, _JsonlHandler)]:
        root.removeHandler(h)
    root.addHandler(_JsonlHandler(log))
    # Lightning's INFO banner lines are unstructured; keep only its warnings.
    logging.getLogger("lightning").setLevel(logging.WARNING)
    logging.getLogger("lightning.pytorch").setLevel(logging.WARNING)
    logging.getLogger("lightning.fabric").setLevel(logging.WARNING)
    return log


def _device_info() -> dict[str, Any]:
    import torch

    info: dict[str, Any] = {"device": "cpu", "gpu_name": None, "gpu_mem_gb": None}
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        info = {"device": "cuda:0", "gpu_name": props.name,
                "gpu_mem_gb": round(props.total_memory / 2**30, 1)}
    return info


def run_header(resolved_cfg: dict[str, Any], **extra: Any) -> dict[str, Any]:
    """Fields for the `run_start` event: makes any log self-describing."""
    import torch

    return {
        "git_sha": git_sha(),
        "git_dirty": git_dirty(),
        "config_hash": config_hash(resolved_cfg),
        "resolved_config": resolved_cfg,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "lightning": L.__version__,
        **_device_info(),
        "cpu_count": os.cpu_count(),
        "host_ram_gb": round(psutil.virtual_memory().total / 2**30, 1),
        **extra,
    }


def failure_fields(exc: BaseException, resolved_cfg: dict[str, Any] | None,
                   step: int | None) -> dict[str, Any]:
    return {
        "exception": f"{type(exc).__name__}: {exc}",
        "traceback": "".join(traceback.format_exception(exc)),
        "step": step,
        "resolved_config": resolved_cfg,
    }


# ---------------------------------------------------------------- live metrics

@functools.cache
def _gpu():
    try:
        import pynvml

        pynvml.nvmlInit()
        return pynvml, pynvml.nvmlDeviceGetHandleByIndex(0)
    except Exception:
        return None


def sys_fields() -> dict[str, Any]:
    """Host RAM/CPU and, when present, GPU utilisation/memory. Sampled by the heartbeat, so
    it runs on the heartbeat interval. Degrades to CPU-only fields without a GPU."""
    vm = psutil.virtual_memory()
    out: dict[str, Any] = {"host_ram_used_gb": round(vm.used / 2**30, 2),
                           "host_ram_frac": round(vm.percent / 100, 3),
                           "cpu_util": round(psutil.cpu_percent(interval=None) / 100, 3)}
    if (gpu := _gpu()) is not None:
        nv, h = gpu
        try:
            out["gpu_util"] = nv.nvmlDeviceGetUtilizationRates(h).gpu / 100
            out["gpu_mem_used_gb"] = round(nv.nvmlDeviceGetMemoryInfo(h).used / 2**30, 2)
        except Exception:
            pass
    return out


class Heartbeat:
    """Emits a `heartbeat` line every `interval_s` of WALL time from a background thread, so
    a hung step still produces lines (with a frozen step) instead of going silent."""

    def __init__(self, log: RunLog, interval_s: float,
                 sources: list[Callable[[], dict[str, Any]]] | None = None):
        self.log, self.interval_s, self.sources = log, interval_s, sources or []
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

    def on_fit_start(self, trainer, pl_module) -> None:
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

    def on_fit_end(self, trainer, pl_module) -> None:
        self.hb.stop()

    def on_exception(self, trainer, pl_module, exception: BaseException) -> None:
        self.hb.stop()


class ThroughputCallback(L.Callback):
    """samples/s and data_wait_frac: the fraction of wall time spent waiting on the
    dataloader (end of one train step -> start of the next). High values mean the job is
    input-bound. Accelerator work is async, so per-step compute time is approximate."""

    WARMUP_BATCHES = 5  # the first fetches include worker startup

    def __init__(self, log: RunLog, warn_data_wait_frac: float = 0.3):
        self.run_log, self.warn_frac = log, warn_data_wait_frac
        self._warned_epoch = -1
        self.on_train_epoch_start()

    def on_train_epoch_start(self, *_) -> None:
        self.batches = self.samples = 0
        self.wait_s = self.compute_s = 0.0
        self._t_batch_start: float | None = None
        self._t_batch_end = time.perf_counter()  # epoch start

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx) -> None:
        self._t_batch_start = now = time.perf_counter()
        self.wait_s += now - self._t_batch_end

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx) -> None:
        self._t_batch_end = now = time.perf_counter()
        if self._t_batch_start is not None:
            self.compute_s += now - self._t_batch_start
        self.samples += len(batch["label"])
        self.batches += 1
        frac = self.fields()["data_wait_frac"]
        if (self.batches >= self.WARMUP_BATCHES and frac > self.warn_frac
                and self._warned_epoch != trainer.current_epoch):
            self._warned_epoch = trainer.current_epoch
            self.run_log.emit("data_wait_high", level="warning", data_wait_frac=frac,
                              threshold=self.warn_frac, epoch=trainer.current_epoch)

    def on_train_epoch_end(self, trainer, pl_module) -> None:
        self.run_log.emit("throughput", phase="train", epoch=trainer.current_epoch,
                          **self.fields())

    def fields(self) -> dict[str, Any]:
        total = self.wait_s + self.compute_s
        return {"samples_per_s": round(self.samples / total, 2) if total > 0 else None,
                "data_wait_frac": round(self.wait_s / total, 4) if total > 0 else None}
