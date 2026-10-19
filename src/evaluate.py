"""Evaluation entrypoint: checkpoint + degradation -> results rows (test split).

  python -m src.evaluate run_dir=outputs/<run_id> degradation=cloud severities=[0,0.5,1]
  python -m src.evaluate run_dir=outputs/<run_id> stage=pilot   # every degradation of a grid

Every evaluation writes its own <run dir>/results/<eval_id>.csv (no locking, so parallel
evaluations of one run never share a file); `src.analysis aggregate` merges them.
"""
import csv
import json
import logging
import sys
import time
from collections.abc import Iterator
from dataclasses import asdict, dataclass, fields
from datetime import datetime, timezone
from pathlib import Path

import hydra
import torch
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torch import Tensor

from src.env import detect
from src.models import LitModule, make_metrics
from src.obs import Heartbeat, failure_fields, new_run_id, run_header, setup_logging, sys_fields

CONFIG_DIR = Path(__file__).resolve().parent.parent / "configs"
RUN_CONFIG = "run_config.yaml"
RESULTS_DIR = "results"


@dataclass
class ResultsRow:
    run_id: str
    git_sha: str
    config_hash: str        # hash of the resolved training config
    timestamp: str
    model: str
    data: str
    degradation: str
    severity: float
    seed: int
    split: str              # degradation sweeps run on "test"
    ckpt_kind: str          # "best" | "last" | explicit path
    threshold: float        # multi-label decision threshold used
    f1_macro: float
    f1_micro: float
    accuracy: float
    ece: float | None
    params: int
    latency_ms: float | None
    flops: float | None = None           # forward FLOPs per sample, clean input
    f1_per_class: str | None = None      # JSON list, one F1 per class
    modality_weights: str | None = None  # JSON {modality: mean fusion weight}; gates/late only


FIELDNAMES = [f.name for f in fields(ResultsRow)]


def append_rows(rows: list[ResultsRow], path: Path) -> None:
    """Append rows to a results CSV, writing the header if the file is new."""
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists()
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if new:
            writer.writeheader()
        writer.writerows(asdict(r) for r in rows)


def push_rows(csv_path: Path, repo_id: str, token: str | None) -> bool:
    """Upload one evaluation's CSV to `results/<run_id>/<eval_id>.csv` in a Hub dataset repo.
    Never raises: a failed push must not kill the run, since the local file is still written."""
    run_id = csv_path.parent.parent.name
    try:
        from huggingface_hub import HfApi

        HfApi(token=token).upload_file(
            path_or_fileobj=str(csv_path), path_in_repo=f"results/{run_id}/{csv_path.name}",
            repo_id=repo_id, repo_type="dataset",
            commit_message=f"results for {run_id}/{csv_path.stem}")
        return True
    except Exception as e:  # network, auth, missing repo, ...
        logging.getLogger(__name__).warning("results push to %s failed: %s", repo_id, e)
        return False


def resolve_ckpt(ckpt: str, run_dir: str | None) -> tuple[Path, Path, str]:
    """Return (ckpt_path, run_dir, ckpt_kind). `ckpt` is "best", "last" or a path."""
    if ckpt in ("best", "last"):
        if not run_dir:
            raise ValueError(f"ckpt={ckpt} needs run_dir")
        return Path(run_dir) / "checkpoints" / f"{ckpt}.ckpt", Path(run_dir), ckpt
    path = Path(ckpt)
    return path, Path(run_dir) if run_dir else path.parent.parent, str(path)


def count_flops(module: torch.nn.Module, batch: dict[str, Tensor | None]) -> float | None:
    """Forward FLOPs per sample on one batch (torch's FlopCounterMode: matmul/conv only)."""
    from torch.utils.flop_counter import FlopCounterMode

    try:
        counter = FlopCounterMode(display=False)
        with counter:
            logits = module(batch)
        return counter.get_total_flops() / logits.shape[0]
    except Exception:  # an op the counter cannot trace must not kill the evaluation
        return None


def degraded_batches(datamodule, degradation, severity: float,
                     degradation_seed: int) -> Iterator[dict[str, Tensor | None]]:
    """Test batches, prepared (standardised, never augmented) and then degraded. A fresh
    generator per severity makes the corruption independent of evaluation order, the model,
    and the global torch RNG."""
    generator = torch.Generator().manual_seed(degradation_seed)
    for batch in datamodule.test_dataloader():
        yield degradation(datamodule.prepare_batch(batch), severity, generator)


def _to(batch: dict, device: torch.device) -> dict:
    return {k: v.to(device) if isinstance(v, Tensor) else v for k, v in batch.items()}


def evaluate(run_dir: str | None, ckpt: str, degradation_cfg: DictConfig,
             degradation_name: str, severities: list[float], degradation_seed: int,
             results_repo: str | None = None, heartbeat_s: float = 30.0,
             jsonl: bool = True) -> list[ResultsRow]:
    """Evaluate one checkpoint at each severity; write `<run dir>/results/<eval_id>.csv`."""
    ckpt_path, rd, ckpt_kind = resolve_ckpt(ckpt, run_dir)
    run = OmegaConf.load(rd / RUN_CONFIG)
    env = detect()
    log = setup_logging(run.meta.run_id, rd, to_file=jsonl)
    eval_id = new_run_id()
    eval_cfg = {"eval_id": eval_id, "ckpt": str(ckpt_path), "ckpt_kind": ckpt_kind,
                "degradation": degradation_name,
                "degradation_cfg": OmegaConf.to_container(degradation_cfg, resolve=True),
                "severities": list(severities), "degradation_seed": degradation_seed}
    log.emit("run_start", entrypoint="evaluate", runtime=str(env.runtime),
             **run_header(eval_cfg))
    heartbeat = Heartbeat(log, heartbeat_s, sources=[sys_fields])
    t0 = time.monotonic()
    try:
        rows = _evaluate(run.meta, run.cfg, ckpt_path, ckpt_kind, degradation_cfg,
                         degradation_name, severities, degradation_seed, log, heartbeat)
    except BaseException as e:
        heartbeat.stop()
        log.emit("run_failed", level="error", **failure_fields(e, eval_cfg, 0))
        log.close()
        raise
    heartbeat.stop()
    results_csv = rd / RESULTS_DIR / f"{eval_id}.csv"
    append_rows(rows, results_csv)
    pushed = None
    if results_repo and env.has_internet:
        pushed = push_rows(results_csv, results_repo, env.secret("HF_TOKEN"))
    log.emit("run_end", status="ok", entrypoint="evaluate", eval_id=eval_id, rows=len(rows),
             results_file=str(results_csv), duration_s=round(time.monotonic() - t0, 2),
             results_pushed=pushed)
    log.close()
    return rows


@torch.no_grad()
def _evaluate(meta, train_cfg, ckpt_path, ckpt_kind, degradation_cfg, degradation_name,
              severities, degradation_seed, log, heartbeat) -> list[ResultsRow]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sync = torch.cuda.synchronize if device.type == "cuda" else lambda: None
    module = LitModule.load_from_checkpoint(ckpt_path, map_location=device).eval()
    threshold = float(module.hparams.threshold)
    num_classes = int(module.hparams.model_cfg["num_classes"])
    params = sum(p.numel() for p in module.parameters())

    datamodule = instantiate(train_cfg.data)
    datamodule.setup("test")
    degradation = instantiate(degradation_cfg)
    degradation.bind(datamodule)
    n_batches = len(datamodule.test_dataloader())
    first = datamodule.prepare_batch(next(iter(datamodule.test_dataloader())))
    flops = count_flops(module, _to(first, device))
    modalities = getattr(module.model, "required_modalities", ())
    heartbeat.update(phase="eval", step=0, max_steps=n_batches * len(severities),
                     degradation=degradation_name)
    heartbeat.start()

    rows = []
    for i, severity in enumerate(severities):
        metrics = make_metrics(num_classes, threshold, per_class=True).to(device)
        elapsed, n, weight_sum = 0.0, 0, None
        # Degrade on CPU (the generator lives there), then move to the model's device.
        for j, batch in enumerate(degraded_batches(datamodule, degradation, severity,
                                                   degradation_seed)):
            batch = _to(batch, device)
            sync()  # time the forward pass, not queued GPU work
            t0 = time.perf_counter()
            logits = module(batch)
            sync()
            elapsed += time.perf_counter() - t0
            n += logits.shape[0]
            metrics.update(logits, batch["label"].int())
            if (w := getattr(module.model, "gate_weights", None)) is not None:
                weight_sum = w.sum(0) if weight_sum is None else weight_sum + w.sum(0)
            heartbeat.update(step=i * n_batches + j + 1, severity=float(severity))
        computed = metrics.compute()
        per_class = [round(float(v), 6) for v in computed.pop("f1_per_class")]
        m = {k: float(v) for k, v in computed.items()}
        weights = (json.dumps({mod: round(float(v) / n, 6)
                               for mod, v in zip(modalities, weight_sum)})
                   if weight_sum is not None and n else None)
        rows.append(ResultsRow(
            run_id=meta.run_id, git_sha=meta.git_sha, config_hash=meta.config_hash,
            timestamp=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            model=meta.model, data=meta.data, degradation=degradation_name,
            severity=float(severity), seed=int(train_cfg.seed), split="test",
            ckpt_kind=ckpt_kind, threshold=threshold, f1_macro=m["f1_macro"],
            f1_micro=m["f1_micro"], accuracy=m["accuracy"], ece=None, params=params,
            latency_ms=1000 * elapsed / n if n else None, flops=flops,
            f1_per_class=json.dumps(per_class), modality_weights=weights))
        log.emit("eval_progress", degradation=degradation_name, severity=float(severity),
                 done=i + 1, total=len(severities), **m)
    return rows


def sweep(cfg: DictConfig) -> int:
    """Evaluate one run under every degradation of `cfg.grids[cfg.stage]`, one results file
    per degradation (so a re-run supersedes cleanly). Returns the number of rows."""
    rows = 0
    for name, severities in cfg.grids[cfg.stage].items():
        degradation_cfg = OmegaConf.load(CONFIG_DIR / "degradation" / f"{name}.yaml")
        rows += len(evaluate(str(cfg.run_dir), str(cfg.ckpt), degradation_cfg, name,
                             [float(s) for s in severities], cfg.degradation_seed,
                             cfg.results_repo, cfg.obs.heartbeat_s, cfg.obs.jsonl))
    return rows


@hydra.main(config_path="../configs", config_name="evaluate", version_base="1.3")
def _hydra_main(cfg: DictConfig) -> None:
    if cfg.stage:
        sweep(cfg)
        return
    evaluate(cfg.run_dir, str(cfg.ckpt), cfg.degradation,
             HydraConfig.get().runtime.choices["degradation"], list(cfg.severities),
             cfg.degradation_seed, cfg.results_repo, cfg.obs.heartbeat_s, cfg.obs.jsonl)


def main() -> None:
    sys.argv = detect().hydra_argv(sys.argv)
    _hydra_main()


if __name__ == "__main__":
    main()
