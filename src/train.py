"""Training entrypoint: config -> checkpoint."""
import sys
import time
from datetime import timedelta
from pathlib import Path

import hydra
import lightning as L
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from lightning.pytorch.callbacks import ModelCheckpoint
from omegaconf import DictConfig, OmegaConf

from src.callbacks.heartbeat import Heartbeat, HeartbeatCallback
from src.callbacks.sysmetrics import SysMetricsCallback
from src.callbacks.throughput import ThroughputCallback
from src.env import Env, detect, get_output_root
from src.module import LitModule
from src.results import RUN_CONFIG
from src.obs import (
    config_hash,
    failure_fields,
    git_sha,
    new_run_id,
    resolve_wandb_mode,
    run_header,
    setup_logging,
)

CLEAN = {"_target_": "src.degradations.none.NoDegradation"}


def _wandb_logger(cfg: DictConfig, env: Env, run_id: str, output_dir: Path):
    from lightning.pytorch.loggers import WandbLogger

    mode = resolve_wandb_mode(cfg.wandb.mode, env.has_internet)
    return WandbLogger(project=cfg.wandb.project, name=run_id, id=run_id,
                       save_dir=str(output_dir), mode=mode, resume="allow")


def run(cfg: DictConfig, choices: dict[str, str]) -> Path:
    """Train once; return the run's durable output dir.

    `choices` maps config group -> selected option (e.g. {"model": "dummy"}), recorded so
    evaluation can label results without re-specifying them. With `resume_from_checkpoint`,
    training continues in the original run dir under the original run_id.
    """
    env = detect()
    resume = cfg.get("resume_from_checkpoint")
    if resume:
        output_dir = Path(resume).resolve().parent.parent
    else:
        output_dir = get_output_root(cfg.get("output_dir")) / new_run_id()
    output_dir.mkdir(parents=True, exist_ok=True)
    run_id = output_dir.name
    log = setup_logging(run_id, output_dir)

    resolved = OmegaConf.to_container(cfg, resolve=True)
    log.emit("run_start", entrypoint="train", runtime=str(env.runtime),
             resumed_from=resume or None, **run_header(resolved))
    if not (output_dir / RUN_CONFIG).exists():
        meta = {
            "run_id": run_id,
            "git_sha": git_sha(),
            "config_hash": config_hash(resolved),
            "model": choices.get("model", "unknown"),
            "data": choices.get("data", "unknown"),
        }
        OmegaConf.save(OmegaConf.create({"meta": meta, "cfg": resolved}), output_dir / RUN_CONFIG)

    t0 = time.monotonic()
    trainer: L.Trainer | None = None
    sysmetrics = SysMetricsCallback()
    throughput = ThroughputCallback(log, cfg.obs.warn_data_wait_frac)
    heartbeat = Heartbeat(log, cfg.obs.heartbeat_s, sources=[throughput.fields, sysmetrics.fields])
    try:
        L.seed_everything(cfg.seed, workers=True, verbose=False)
        datamodule = instantiate(cfg.data)
        module = LitModule(
            model_cfg=resolved["model"], optim_cfg=resolved["optim"], threshold=cfg.threshold
        )
        ckpt_dir = output_dir / "checkpoints"
        best_cb = ModelCheckpoint(
            dirpath=ckpt_dir, filename="best", monitor="val/f1_macro", mode="max", save_last=True
        )
        callbacks = [
            best_cb,
            # Wall-clock checkpoints so a run killed by a runtime time cap can resume.
            ModelCheckpoint(
                dirpath=ckpt_dir,
                filename="interval",
                train_time_interval=timedelta(minutes=cfg.ckpt_interval_min),
                enable_version_counter=False,
            ),
            HeartbeatCallback(heartbeat),
            throughput,
            sysmetrics,
        ]
        trainer_kwargs = dict(cfg.trainer)
        if cfg.wandb.enabled:
            trainer_kwargs["logger"] = _wandb_logger(cfg, env, run_id, output_dir)
        trainer = L.Trainer(**trainer_kwargs, callbacks=callbacks, default_root_dir=output_dir)
        trainer.fit(module, datamodule=datamodule, ckpt_path=resume or None)
    except BaseException as e:
        heartbeat.stop()
        step = trainer.global_step if trainer is not None else None
        log.emit("run_failed", level="error", **failure_fields(e, resolved, step))
        log.close()
        raise
    heartbeat.stop()
    score = best_cb.best_model_score
    log.emit("run_end", status="ok", duration_s=round(time.monotonic() - t0, 2),
             global_step=trainer.global_step, best_ckpt=best_cb.best_model_path or None,
             best_val_f1_macro=float(score) if score is not None else None)
    log.close()

    # One clean (undegraded) test-split row per training run, from the best checkpoint.
    from src.evaluate import evaluate

    evaluate(str(output_dir), "best", OmegaConf.create(CLEAN), "none", [0.0],
             cfg.degradation_seed, results_repo=cfg.get("results_repo"),
             heartbeat_s=cfg.obs.heartbeat_s)
    return output_dir


@hydra.main(config_path="../configs", config_name="config", version_base="1.3")
def _hydra_main(cfg: DictConfig) -> None:
    run(cfg, dict(HydraConfig.get().runtime.choices))


def main() -> None:
    sys.argv = detect().hydra_argv(sys.argv)
    _hydra_main()


if __name__ == "__main__":
    main()
