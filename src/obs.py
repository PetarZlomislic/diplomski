"""Observability and provenance. Full JSONL logging arrives in Phase 6."""
import hashlib
import json
import secrets
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent


def new_run_id() -> str:
    """Timestamp + 6 hex chars, generated once per run."""
    return datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S") + "-" + secrets.token_hex(3)


def git_sha() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True, check=True
        )
        return out.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def config_hash(resolved_cfg: dict[str, Any]) -> str:
    blob = json.dumps(resolved_cfg, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:12]


def setup_logging(output_dir: Path) -> None:
    """Placeholder until Phase 6."""
