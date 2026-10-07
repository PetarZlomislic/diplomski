"""Training entrypoint: config -> checkpoint."""
from pathlib import Path

import hydra
import lightning as L
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from lightning.pytorch.callbacks import ModelCheckpoint
from omegaconf import DictConfig, OmegaConf

from src.env import get_output_root
from src.module import LitModule
from src.obs import config_hash, git_sha, new_run_id, setup_logging

RUN_CONFIG = "run_config.yaml"


def run(cfg: DictConfig, choices: dict[str, str]) -> Path:
    """Train once; return the run's durable output dir.

    `choices` maps config group -> selected option (e.g. {"model": "dummy"}), recorded so
    evaluation can label results without re-specifying them.
    """
    run_id = new_run_id()
    output_dir = get_output_root(cfg) / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(output_dir)

    resolved = OmegaConf.to_container(cfg, resolve=True)
    meta = {
        "run_id": run_id,
        "git_sha": git_sha(),
        "config_hash": config_hash(resolved),
        "model": choices.get("model", "unknown"),
        "data": choices.get("data", "unknown"),
    }
    OmegaConf.save(OmegaConf.create({"meta": meta, "cfg": resolved}), output_dir / RUN_CONFIG)

    L.seed_everything(cfg.seed, workers=True)
    datamodule = instantiate(cfg.data)
    module = LitModule(
        model_cfg=resolved["model"], optim_cfg=resolved["optim"], threshold=cfg.threshold
    )
    ckpt_cb = ModelCheckpoint(
        dirpath=output_dir / "checkpoints",
        filename="best",
        monitor="val/f1_macro",
        mode="max",
        save_last=True,
    )
    trainer = L.Trainer(**cfg.trainer, callbacks=[ckpt_cb], default_root_dir=output_dir)
    trainer.fit(module, datamodule=datamodule)
    return output_dir


@hydra.main(config_path="../configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    run(cfg, dict(HydraConfig.get().runtime.choices))


if __name__ == "__main__":
    main()
