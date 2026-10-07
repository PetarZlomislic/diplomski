"""Training entrypoint: config -> checkpoint."""
import sys
from datetime import timedelta
from pathlib import Path

import hydra
import lightning as L
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from lightning.pytorch.callbacks import ModelCheckpoint
from omegaconf import DictConfig, OmegaConf

from src.env import detect, get_output_root
from src.module import LitModule
from src.obs import config_hash, git_sha, new_run_id, setup_logging

RUN_CONFIG = "run_config.yaml"


def run(cfg: DictConfig, choices: dict[str, str]) -> Path:
    """Train once; return the run's durable output dir.

    `choices` maps config group -> selected option (e.g. {"model": "dummy"}), recorded so
    evaluation can label results without re-specifying them. With `resume_from_checkpoint`,
    training continues in the original run dir under the original run_id.
    """
    resume = cfg.get("resume_from_checkpoint")
    if resume:
        output_dir = Path(resume).resolve().parent.parent
    else:
        output_dir = get_output_root(cfg.get("output_dir")) / new_run_id()
    output_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(output_dir)

    resolved = OmegaConf.to_container(cfg, resolve=True)
    if not (output_dir / RUN_CONFIG).exists():
        meta = {
            "run_id": output_dir.name,
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
    ckpt_dir = output_dir / "checkpoints"
    callbacks = [
        ModelCheckpoint(
            dirpath=ckpt_dir, filename="best", monitor="val/f1_macro", mode="max", save_last=True
        ),
        # Wall-clock checkpoints so a run killed by a runtime time cap can resume.
        ModelCheckpoint(
            dirpath=ckpt_dir,
            filename="interval",
            train_time_interval=timedelta(minutes=cfg.ckpt_interval_min),
            enable_version_counter=False,
        ),
    ]
    trainer = L.Trainer(**cfg.trainer, callbacks=callbacks, default_root_dir=output_dir)
    trainer.fit(module, datamodule=datamodule, ckpt_path=resume or None)
    return output_dir


@hydra.main(config_path="../configs", config_name="config", version_base="1.3")
def _hydra_main(cfg: DictConfig) -> None:
    run(cfg, dict(HydraConfig.get().runtime.choices))


def main() -> None:
    sys.argv = detect().hydra_argv(sys.argv)
    _hydra_main()


if __name__ == "__main__":
    main()
