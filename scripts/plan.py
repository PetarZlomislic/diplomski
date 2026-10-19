"""Plan a remote job: predict its wall time and cost on every option usable right now, and pick
the best trade-off between them. Used by `submit.py auto`, which then submits to the pick.

The pick minimises  cost + time_value x hours  (time_value: what an hour of waiting is worth
to you, `--time-value`, default $1/h; 0 = cheapest, large = fastest), among options that
finish before `--deadline` and fit the job (GPU, disk, RAM, Kaggle's session cap and quota).

Wall time = overhead + dataset download + per run (setup + training + evaluation). Per sample,
waiting for data and computing add up: 1/rate = 1/gpu_rate + 1/loader_rate. The pilot jobs
were input-bound (data_wait 0.4-0.7), so vCPUs matter as much as the GPU. Calibration, from
the Experiment 1 pilot jobs of 2026-10-13 (a10g-large: 12 vCPU, 10 workers; a10g-small: 4 vCPU,
4 workers; width 16, 120 px), which this reproduces within ~10%:
  * compute-only A10G training rate per arm = samples_per_s / (1 - data_wait_frac)
  * loader ~315 samples/s per worker, workers capped at vCPUs - 2 (main process + GPU feed)
  * dataset download ~0.035 GB/s per vCPU (41.5 GB: ~100 s on 12 vCPU, ~259 s on 4)
  * bootstrap ~30 s on HF Jobs, ~150 s on Kaggle (installing uv + sync); ~12 s setup per run
GPU speed relative to the A10G is an ASSUMPTION from public small-CNN training benchmarks,
not a measurement. Model FLOPs are scaled as (width/16)^2 x (image/120)^2. Entries without a
model (prepare_data) are priced at --timeout.
"""
import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REPO = r"[A-Za-z0-9._-]+/[A-Za-z0-9._-]+"

A10G_TRAIN_SPS = {"opt": 4040, "sar": 4270, "early": 3990, "late": 1730, "gate_global": 1750,
                  "gate_spatial": 2180}
DEFAULT_TRAIN_SPS = 2000  # arms not in the calibration
EVAL_SPEEDUP = 3.0  # forward only
CPU_FACTOR = 0.03  # training on CPU instead of an A10G
PER_WORKER_SPS = 315
WORKER_RESERVED_VCPU = 2
BW_PER_VCPU, BW_MAX = 0.035, 1.0  # GB/s
RUN_SETUP_S = 12
GPU_FACTOR = {"T4": 0.4, "P100": 0.45, "L4": 0.75, "A10G": 1.0, "L40S": 2.0, "A100": 2.2,
              "RTX PRO 6000": 2.6, "H100": 3.0, "H200": 3.2}
KAGGLE = {"vcpu": 4, "gpu": "T4", "disk_gb": 50, "gpu_h": 9, "cpu_h": 12, "overhead_s": 150}
HF_OVERHEAD_S = 30
SAFETY = 1.25  # predicted hours x this must fit caps and quota


@dataclass
class RunWork:
    arm: str | None
    width: int = 16
    image: int = 120
    train: int = 0  # training samples over all epochs
    evals: int = 0  # evaluated samples (validation passes + test sweeps)
    workers: int = 0
    gpu: bool = True


@dataclass
class Work:
    jobs: list[list[RunWork] | None]  # the runs of each job; None = no speed model
    data_gb: float  # downloaded by every job

    @property
    def modelled(self) -> bool:
        return all(j is not None for j in self.jobs)

    @property
    def gpu(self) -> bool:
        return any(r.gpu for j in self.jobs if j for r in j)


@dataclass
class Option:
    name: str
    price_h: float
    vcpu: int
    gpu: str | None
    disk_gb: float
    ram_gb: float
    overhead_s: float
    why_not: str | None = None
    hours: float = 0.0
    cost: float = 0.0
    loader_bound: bool = False
    tags: list[str] = field(default_factory=list)


# ---------------------------------------------------------------- the job

def _compose(name: str, overrides: list[str]):
    from hydra import compose, initialize_config_dir

    with initialize_config_dir(config_dir=str(ROOT / "configs"), version_base="1.3"):
        return compose(name, overrides=overrides, return_hydra_config=True)


def _sizes(data, api) -> dict[str, int]:
    """Samples per split: the shard manifest (Hub repo or local dir), or the dummy split."""
    if "data_uri" not in data or not str(data.get("data_uri") or ""):
        n = int(data.get("n_samples", 64))
        val = max(1, n // 5)
        return {"train": max(1, n - 2 * val), "val": val, "test": val}
    uri = str(data.data_uri)
    for local in (Path(uri), ROOT / "data" / uri):
        if (local / "manifest.json").exists():
            return json.loads((local / "manifest.json").read_text())["sizes"]
    if re.fullmatch(REPO, uri):
        path = api.hf_hub_download(uri, "manifest.json", repo_type="dataset")
        return json.loads(Path(path).read_text())["sizes"]
    raise ValueError(f"cannot find the manifest of {uri}")


def _run(cfg, api, test_passes: int) -> RunWork:
    sizes = _sizes(cfg.data, api)
    epochs = int(cfg.trainer.max_epochs)
    enc = cfg.model.get("encoder") or {}
    return RunWork(arm=cfg.hydra.runtime.choices.get("model"), width=int(enc.get("width", 16)),
                   image=int(cfg.data.image_size), train=epochs * sizes["train"],
                   evals=epochs * sizes.get("val", 0) + test_passes * sizes.get("test", 0),
                   workers=int(cfg.data.get("num_workers", 0)),
                   gpu=str(cfg.trainer.get("accelerator", "auto")) != "cpu")


def job_runs(entry: str, overrides: list[str], data_repos: list[str],
             api) -> list[RunWork] | None:
    """What one job does, from its own config: runs, epochs, samples, workers."""
    from omegaconf import OmegaConf

    if entry == "train":
        return [_run(_compose("config", overrides), api, 1)]
    if entry == "exp1_run":
        args = OmegaConf.from_dotlist(overrides)
        stage = str(args.get("stage", "pilot"))
        grid = _compose("evaluate", []).grids.get(stage, {}) if stage != "none" else {}
        sweep = sum(len(v) for v in grid.values())
        extra = [str(o) for o in args.get("train", [])]
        return [_run(_compose("config", [*extra, f"model={m}", f"seed={s}"]), api, 1 + sweep)
                for s in args.get("seeds", [0]) for m in args.get("models", [])]
    if entry == "evaluate":
        cfg = _compose("evaluate", overrides)
        n = (sum(len(v) for v in cfg.grids[cfg.stage].values()) if cfg.stage
             else len(cfg.severities))
        test = _sizes(OmegaConf.create({"data_uri": data_repos[0]}), api)["test"] \
            if data_repos else 20000
        return [RunWork(arm=None, evals=n * test, workers=4)]
    return None


# ---------------------------------------------------------------- the options

def _gb(spec: str | None) -> float:
    m = re.match(r"([\d.]+)\s*([GT])B", spec or "")
    return float(m[1]) * (1000 if m[2] == "T" else 1) if m else 0.0


def hf_options(api) -> list[Option]:
    """Every single-device HF Jobs flavor with its live price."""
    out = []
    for h in api.list_jobs_hardware():
        acc = h.accelerator
        if acc is not None and int(acc.quantity) > 1:
            continue  # training uses one device
        out.append(Option(h.name, h.unit_cost_usd * 60, int(re.match(r"\d+", h.cpu)[0]),
                          acc.model if acc else None, _gb(h.ephemeral_storage), _gb(h.ram),
                          HF_OVERHEAD_S))
    return out


def kaggle_option() -> Option:
    return Option("kaggle", 0.0, KAGGLE["vcpu"], KAGGLE["gpu"], KAGGLE["disk_gb"], 30,
                  KAGGLE["overhead_s"])


def usable_workers(workers: int, vcpu: int) -> int:
    return max(1, min(workers, vcpu - WORKER_RESERVED_VCPU))


def job_hours(runs: list[RunWork] | None, data_gb: float, o: Option, timeout_h: float) -> float:
    if runs is None:
        return timeout_h
    s = o.overhead_s + data_gb / min(BW_MAX, BW_PER_VCPU * o.vcpu)
    for r in runs:
        flops = (r.width / 16) ** 2 * (r.image / 120) ** 2
        gpu = A10G_TRAIN_SPS.get(r.arm, DEFAULT_TRAIN_SPS) / flops * (
            GPU_FACTOR.get(o.gpu, 1.0) if o.gpu and r.gpu else CPU_FACTOR)
        loader = PER_WORKER_SPS * usable_workers(r.workers, o.vcpu) * 120**2 / r.image**2
        o.loader_bound |= loader < gpu
        s += (RUN_SETUP_S + r.train * (1 / gpu + 1 / loader)
              + r.evals * (1 / (gpu * EVAL_SPEEDUP) + 1 / loader))
    return s / 3600


def predict(w: Work, o: Option, timeout_h: float) -> None:
    """o.hours: wall time until the slowest job ends (a sweep's jobs run in parallel);
    o.cost: all jobs; o.loader_bound: data loading, not the GPU, limits some run."""
    hours = [job_hours(j, w.data_gb, o, timeout_h) for j in w.jobs]
    o.hours, o.cost = max(hours), o.price_h * sum(hours)


def plan(w: Work, options: list[Option], time_value: float, deadline_h: float | None,
         timeout_h: float, need_disk: float, need_ram: float) -> list[Option]:
    """Options ranked best first; unusable ones last, with why_not set."""
    for o in options:
        predict(w, o, timeout_h)
        if o.why_not:
            continue
        if w.gpu and o.gpu is None:
            o.why_not = "no GPU"
        elif o.disk_gb < need_disk:
            o.why_not = f"{o.disk_gb:.0f} GB disk < {need_disk:.0f} GB needed"
        elif o.ram_gb < need_ram:
            o.why_not = f"{o.ram_gb:.0f} GB RAM < {need_ram:.0f} GB needed"
        elif deadline_h is not None and o.hours > deadline_h:
            o.why_not = f"~{fmt_h(o.hours)} > deadline {fmt_h(deadline_h)}"
    ok = sorted((o for o in options if not o.why_not),
                key=lambda o: (o.cost + time_value * o.hours, o.hours))
    for o in ok:  # Pareto: nothing else is both cheaper and faster
        if not any(p.cost <= o.cost and p.hours < o.hours or p.cost < o.cost
                   and p.hours <= o.hours for p in ok):
            o.tags.append("pareto")
    if ok:
        ok[0].tags.insert(0, "pick")
        min(ok, key=lambda o: (o.cost, o.hours)).tags.append("cheapest")
        min(ok, key=lambda o: (o.hours, o.cost)).tags.append("fastest")
    return ok + [o for o in options if o.why_not]


def suggested_timeout(hours: float) -> str:
    """Generous cap for the job: 1.5x the prediction plus 15 min, in whole minutes."""
    return f"{math.ceil((hours * 1.5 + 0.25) * 60)}m"


def fmt_h(hours: float) -> str:
    return f"{hours * 60:.0f} min" if hours < 1 else f"{hours:.1f} h"
