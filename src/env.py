"""Runtime adapter. Runtime detection arrives in Phase 5."""
from pathlib import Path

from omegaconf import DictConfig


def get_output_root(cfg: DictConfig) -> Path:
    """Durable output root. Phase 5 resolves this per runtime."""
    root = Path(cfg.output_dir) if cfg.get("output_dir") else Path("outputs")
    root = root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    return root
