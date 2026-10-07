"""Observability. Full JSONL logging arrives in Phase 6."""
import secrets
from datetime import datetime, timezone
from pathlib import Path


def new_run_id() -> str:
    """Timestamp + 6 hex chars, generated once per run."""
    return datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S") + "-" + secrets.token_hex(3)


def setup_logging(output_dir: Path) -> None:
    """Placeholder until Phase 6."""
