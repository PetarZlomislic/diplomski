"""Observability: structured JSONL to stdout + <run dir>/run.jsonl, and run provenance.

Every line: {"ts", "level", "event", "run_id", ...event fields}. Remote jobs are detached,
so this log is the only window into a run.
"""
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
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TextIO

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


def resolve_wandb_mode(requested: str, has_internet: bool) -> str:
    """Offline instead of hanging when the runtime has no internet."""
    return requested if has_internet else "offline"


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
    import lightning
    import psutil
    import torch

    return {
        "git_sha": git_sha(),
        "git_dirty": git_dirty(),
        "config_hash": config_hash(resolved_cfg),
        "resolved_config": resolved_cfg,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "lightning": lightning.__version__,
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
