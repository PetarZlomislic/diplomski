"""Evaluation entrypoint: checkpoint + degradation -> results rows (test split)."""
import sys
import time
from collections.abc import Iterator
from datetime import datetime, timezone
from pathlib import Path

import hydra
import torch
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from lightning import LightningDataModule
from omegaconf import DictConfig, OmegaConf
from torch import Tensor

from src.callbacks.heartbeat import Heartbeat
from src.callbacks.sysmetrics import SysMetricsCallback
from src.degradations.base import BaseDegradation
from src.env import detect
from src.module import LitModule, make_metrics
from src.obs import failure_fields, run_header, setup_logging
from src.results import ResultsRow, append_rows, push_rows
from src.train import RUN_CONFIG

RESULTS_FILE = "results.csv"


def resolve_ckpt(ckpt: str, run_dir: str | None) -> tuple[Path, Path, str]:
    """Return (ckpt_path, run_dir, ckpt_kind). `ckpt` is "best", "last" or a path."""
    if ckpt in ("best", "last"):
        if not run_dir:
            raise ValueError(f"ckpt={ckpt} needs run_dir")
        rd = Path(run_dir)
        return rd / "checkpoints" / f"{ckpt}.ckpt", rd, ckpt
    path = Path(ckpt)
    return path, Path(run_dir) if run_dir else path.parent.parent, str(path)


def degraded_batches(
    datamodule: LightningDataModule,
    degradation: BaseDegradation,
    severity: float,
    degradation_seed: int,
) -> Iterator[dict[str, Tensor | None]]:
    """Test batches with the degradation applied. A fresh generator per severity makes the
    corruption independent of evaluation order, the model, and the global torch RNG."""
    generator = torch.Generator().manual_seed(degradation_seed)
    for batch in datamodule.test_dataloader():
        yield degradation(batch, severity, generator)


def evaluate(
    run_dir: str | None,
    ckpt: str,
    degradation_cfg: DictConfig,
    degradation_name: str,
    severities: list[float],
    degradation_seed: int,
    results_repo: str | None = None,
    heartbeat_s: float = 30.0,
) -> list[ResultsRow]:
    ckpt_path, rd, ckpt_kind = resolve_ckpt(ckpt, run_dir)
    run = OmegaConf.load(rd / RUN_CONFIG)
    meta, train_cfg = run.meta, run.cfg
    env = detect()
    log = setup_logging(meta.run_id, rd)
    eval_cfg = {"ckpt": str(ckpt_path), "ckpt_kind": ckpt_kind, "degradation": degradation_name,
                "degradation_cfg": OmegaConf.to_container(degradation_cfg, resolve=True),
                "severities": list(severities), "degradation_seed": degradation_seed}
    log.emit("run_start", entrypoint="evaluate", runtime=str(env.runtime),
             **run_header(eval_cfg))
    heartbeat = Heartbeat(log, heartbeat_s, sources=[SysMetricsCallback().fields])
    t0 = time.monotonic()
    done = 0
    try:
        rows = _evaluate(meta, train_cfg, ckpt_path, ckpt_kind, degradation_cfg,
                         degradation_name, severities, degradation_seed, log, heartbeat)
        done = len(rows)
    except BaseException as e:
        heartbeat.stop()
        log.emit("run_failed", level="error", **failure_fields(e, eval_cfg, done))
        log.close()
        raise
    heartbeat.stop()
    append_rows(rows, rd / RESULTS_FILE)
    pushed = None
    if results_repo and env.has_internet:
        pushed = push_rows(rd / RESULTS_FILE, results_repo, env.secret("HF_TOKEN"))
    log.emit("run_end", status="ok", entrypoint="evaluate", rows=len(rows),
             duration_s=round(time.monotonic() - t0, 2), results_pushed=pushed)
    log.close()
    return rows


@torch.no_grad()
def _evaluate(meta, train_cfg, ckpt_path, ckpt_kind, degradation_cfg, degradation_name,
              severities, degradation_seed, log, heartbeat) -> list[ResultsRow]:
    module = LitModule.load_from_checkpoint(ckpt_path, map_location="cpu")
    module.eval()
    threshold = float(module.hparams.threshold)
    num_classes = int(module.hparams.model_cfg["num_classes"])
    params = sum(p.numel() for p in module.parameters())

    datamodule = instantiate(train_cfg.data)
    datamodule.setup("test")
    degradation = instantiate(degradation_cfg)
    n_batches = len(datamodule.test_dataloader())
    heartbeat.update(phase="eval", step=0, max_steps=n_batches * len(severities),
                     degradation=degradation_name)
    heartbeat.start()

    rows = []
    for i, severity in enumerate(severities):
        metrics = make_metrics(num_classes, threshold)
        elapsed, n = 0.0, 0
        batches = degraded_batches(datamodule, degradation, severity, degradation_seed)
        for j, batch in enumerate(batches):
            t0 = time.perf_counter()
            logits = module(batch)
            elapsed += time.perf_counter() - t0
            n += logits.shape[0]
            metrics.update(logits, batch["label"].int())
            heartbeat.update(step=i * n_batches + j + 1, severity=float(severity))
        m = {k: float(v) for k, v in metrics.compute().items()}
        rows.append(
            ResultsRow(
                run_id=meta.run_id,
                git_sha=meta.git_sha,
                config_hash=meta.config_hash,
                timestamp=datetime.now(timezone.utc).isoformat(timespec="seconds"),
                model=meta.model,
                data=meta.data,
                degradation=degradation_name,
                severity=float(severity),
                seed=int(train_cfg.seed),
                split="test",
                ckpt_kind=ckpt_kind,
                threshold=threshold,
                f1_macro=m["f1_macro"],
                f1_micro=m["f1_micro"],
                accuracy=m["accuracy"],
                ece=None,
                params=params,
                latency_ms=1000 * elapsed / n if n else None,
            )
        )
        log.emit("eval_progress", degradation=degradation_name, severity=float(severity),
                 done=i + 1, total=len(severities), **m)
    return rows


@hydra.main(config_path="../configs", config_name="evaluate", version_base="1.3")
def _hydra_main(cfg: DictConfig) -> None:
    evaluate(
        run_dir=cfg.run_dir,
        ckpt=str(cfg.ckpt),
        degradation_cfg=cfg.degradation,
        degradation_name=HydraConfig.get().runtime.choices["degradation"],
        severities=list(cfg.severities),
        degradation_seed=cfg.degradation_seed,
        results_repo=cfg.results_repo,
        heartbeat_s=cfg.obs.heartbeat_s,
    )


def main() -> None:
    sys.argv = detect().hydra_argv(sys.argv)
    _hydra_main()


if __name__ == "__main__":
    main()
