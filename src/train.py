"""Training entrypoint: config -> checkpoint."""
from pathlib import Path

import hydra
import lightning as L
from hydra.utils import instantiate
from lightning.pytorch.callbacks import ModelCheckpoint
from omegaconf import DictConfig, OmegaConf

from src.env import get_output_root
from src.module import LitModule
from src.obs import new_run_id, setup_logging


def run(cfg: DictConfig) -> Path:
    """Train once; return the run's durable output dir."""
    run_id = new_run_id()
    output_dir = get_output_root(cfg) / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(output_dir)

    L.seed_everything(cfg.seed, workers=True)
    datamodule = instantiate(cfg.data)
    module = LitModule(
        model_cfg=OmegaConf.to_container(cfg.model, resolve=True),
        optim_cfg=OmegaConf.to_container(cfg.optim, resolve=True),
    )
    ckpt_cb = ModelCheckpoint(
        dirpath=output_dir / "checkpoints",
        filename="best",
        monitor="val/loss",
        mode="min",
        save_last=True,
    )
    trainer = L.Trainer(**cfg.trainer, callbacks=[ckpt_cb], default_root_dir=output_dir)
    trainer.fit(module, datamodule=datamodule)
    return output_dir


@hydra.main(config_path="../configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    run(cfg)


if __name__ == "__main__":
    main()
