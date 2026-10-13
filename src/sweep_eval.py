"""Experiment 1: evaluate one trained run under every degradation of a stage's grid.

  python -m src.sweep_eval run_dir=outputs/<run_id> stage=pilot
  ./scripts/submit_hf.sh --entry sweep_eval run_dir=<bucket mount>/<run_id> stage=full

One results file per degradation (via src.evaluate), so a re-run supersedes cleanly.
"""
import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

from src.env import detect
from src.evaluate import evaluate

CONFIG_DIR = Path(__file__).resolve().parent.parent / "configs"


def sweep(cfg: DictConfig) -> int:
    grid = cfg.grids[cfg.stage]
    rows = 0
    for name, severities in grid.items():
        degradation_cfg = OmegaConf.load(CONFIG_DIR / "degradation" / f"{name}.yaml")
        rows += len(evaluate(str(cfg.run_dir), str(cfg.ckpt), degradation_cfg, name,
                             [float(s) for s in severities], cfg.degradation_seed,
                             results_repo=cfg.results_repo, heartbeat_s=cfg.obs.heartbeat_s,
                             jsonl=cfg.obs.jsonl))
    return rows


@hydra.main(config_path="../configs", config_name="sweep_eval", version_base="1.3")
def _hydra_main(cfg: DictConfig) -> None:
    sweep(cfg)


def main() -> None:
    sys.argv = detect().hydra_argv(sys.argv)
    _hydra_main()


if __name__ == "__main__":
    main()
