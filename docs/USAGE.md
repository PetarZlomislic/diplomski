# Usage guide

How to run, extend and operate `robustness-eval`. The README is the 40-line quickstart; this is the full manual.

- [1. Setup](#1-setup)
- [2. How a run is organised](#2-how-a-run-is-organised)
- [3. Training](#3-training)
- [4. Evaluation under degradation](#4-evaluation-under-degradation)
- [5. Aggregation and plots](#5-aggregation-and-plots)
- [6. Data preparation and streaming](#6-data-preparation-and-streaming)
- [7. Running on HF Jobs](#7-running-on-hf-jobs)
- [8. Running on Kaggle](#8-running-on-kaggle)
- [9. Reading the logs](#9-reading-the-logs)
- [10. Extending the framework](#10-extending-the-framework)
- [11. Development workflow](#11-development-workflow)
- [12. Troubleshooting](#12-troubleshooting)
- [13. Reference](#13-reference)

---

## 1. Setup

```bash
uv sync --extra dev --extra logging    # Python 3.12 + exact versions from uv.lock
```

`--extra logging` adds W&B; `--extra dev` adds pytest and ruff. On Windows, if the venv was created from a conda Python, rebuild it with `uv venv --python 3.12 --python-preference only-managed`: conda ships an old VC++ runtime that stops torch from loading.

**Secrets** live in a gitignored `.env` at the repo root (copy `.env.example`). The submission scripts load it; variables already set in your shell take precedence.

| Variable | Used by | What |
|---|---|---|
| `HF_TOKEN` | HF Jobs, Hub pushes | Write token (huggingface.co → Settings → Access Tokens) |
| `HF_BUCKET` | `submit_hf.sh` | `<user>/<bucket>`, mounted at `/ckpt` in the job |
| `KAGGLE_API_TOKEN` | `submit_kaggle.sh` | `KGAT_...` token (kaggle.com → Settings → API) |
| `KAGGLE_KERNEL` | `submit_kaggle.sh` | `<kaggle-user>/<kernel-slug>` |
| `WANDB_API_KEY` | optional | Forwarded to jobs only when set |

**CLIs for remote runs:** `hf` comes with the project venv. Install Kaggle's once: `uv tool install kaggle`.

## 2. How a run is organised

Every training run gets a `run_id` (`YYYYMMDD-HHMMSS-<6 hex>`) and a **durable run directory** `<output_root>/<run_id>/`:

```
<run_id>/
├── run_config.yaml        # resolved config + meta (run_id, git_sha, config_hash, model, data)
├── run.jsonl              # structured log of training and every evaluation of this run
├── results/<eval_id>.csv  # one file per evaluation, one row per severity
└── checkpoints/
    ├── best.ckpt          # best val/f1_macro
    ├── last.ckpt          # end of last epoch (or the time-budget stop point)
    └── interval.ckpt      # wall-clock checkpoint every ckpt_interval_min
```

`output_root` depends on where the run happens: `./outputs` locally, `/kaggle/working` on Kaggle, `/ckpt` (your bucket) on HF Jobs. Override it with `output_dir=<path>`. Hydra's own per-run working files go to `hydra_runs/` and are not needed afterwards.

Configuration is Hydra. Pick components with `group=option`, override any value with `key=value`:

```bash
python -m src.train model=dummy_wide data=dummy optim.lr=3e-4 trainer.max_epochs=20 seed=1
```

## 3. Training

```bash
uv run python -m src.train +experiment=smoke                 # tiny CPU run, < 1 min
uv run python -m src.train model=dummy_wide trainer.max_epochs=20
uv run python -m src.train --multirun model=dummy,dummy_wide seed=0,1,2    # 6 runs
uv run python -m src.train --multirun model=dummy model/fusion=passthrough
```

What one training run does:
1. Writes `run_config.yaml` and a `run_start` log line (git SHA, dirty flag, config hash, hardware).
2. Trains with `BCEWithLogitsLoss` (multi-label; models return raw logits).
3. Selects `best.ckpt` on `val/f1_macro`, so a degraded metric is never used for selection.
4. Finally evaluates `best.ckpt` (or `last.ckpt` if no validation ever ran) once on the **test** split without degradation, as its own results file.

**Resume** a run (continues in the same run directory and `run_id`):

```bash
uv run python -m src.train resume_from_checkpoint=outputs/<run_id>/checkpoints/last.ckpt trainer.max_epochs=40
```

**Time caps.** When the runtime has a wall-clock limit (Kaggle 9 h; HF Jobs `--timeout`), training stops `stop_margin_min` (default 15) minutes early, saves `last.ckpt`, and logs a `time_budget_stop` event that contains the exact resume override.

**W&B** is off by default; turn it on with `wandb.enabled=true` (needs `--extra logging`). Without internet it falls back to offline mode and writes `<run_id>/wandb/`. The JSONL log is complete either way.

## 4. Evaluation under degradation

```bash
uv run python -m src.evaluate run_dir=outputs/<run_id> degradation=gaussian_noise
uv run python -m src.evaluate run_dir=outputs/<run_id> degradation=gaussian_noise \
    "severities=[0.0,0.1,0.2,0.4,0.8]" ckpt=last degradation.modality=sar
uv run python -m src.evaluate ckpt=path/to/model.ckpt degradation=none "severities=[0.0]"
```

- The model is rebuilt from the checkpoint; you never pass `model=` again.
- `ckpt` is `best` (default), `last`, or a path. The choice is recorded in each row as `ckpt_kind`.
- Degradations run on the **test** split only, never during training.
- Noise comes from its own generator, seeded only by `degradation_seed` (default 12345) and restarted per severity. Severity 0.8 is therefore bit-identical for every model and every training seed, which is what makes cross-model comparisons valid.
- `severity=0.0` is an exact identity, so that row equals a clean evaluation.
- Each call writes `<run_id>/results/<eval_id>.csv` (one row per severity) and logs an `eval_progress` line per severity. Separate files mean evaluations of the same run can run in parallel, even as remote jobs sharing a bucket.
- On a GPU machine evaluation runs on the GPU; `latency_ms` is measured with CUDA synchronization.

Re-running an evaluation adds a new file. `aggregate` keeps only the latest row per (run, degradation, severity, checkpoint, threshold).

**Pushing rows to the Hub:** set `results_repo=<user>/<dataset>` (on `train` or `evaluate`). When the runtime has internet, each evaluation's file is uploaded to `results/<run_id>/<eval_id>.csv` in that dataset repo. A failed push only logs a warning; the local file is always written.

## 5. Aggregation and plots

```bash
uv run python -m src.aggregate outputs                       # searches outputs/ recursively
uv run python -m src.aggregate outputs --metric f1_micro --out reports/
```

It reads every `<run>/results/*.csv` below the folder and writes `<out>/results_all.csv`, the merged table that is the source for every figure. It also writes one `curve_<degradation>_<metric>.png` per degradation: severity on x, metric on y, one line per model. A model keeps the same color in every figure. To include remote runs, download them first (§7, §8) and point `aggregate` at a folder containing them all.

## 6. Data preparation and streaming

The real dataset (reBEN, ~118 GB) is never downloaded inside a training job. It is prepared **once** into sharded tiers:

```bash
uv run python -m src.prepare_data --source synthetic --fraction 0.01        # one tier
uv run python -m src.prepare_data --source synthetic                        # dev, sweep, full
uv run python -m src.prepare_data --source synthetic --out-repo <user>/reben  # + push tiers
```

| Tier | Fraction | Pushed to |
|---|---|---|
| `dev` | 1 % | `<out-repo>-dev` |
| `sweep` | 15 % | `<out-repo>-sweep` |
| `full` | 100 % | `<out-repo>-full` |

Two guarantees are enforced:
- Patches are sampled **within** each official train/val/test split, never across them. Crossing splits would leak test geography. `--no-split-aware` is refused.
- Every class keeps at least `--min-per-class` patches in every split (rarest classes are filled first), or preparation fails with a `ClassFloorError` naming the short classes.

All 14 bands (12 optical + 2 SAR) are kept in every tier. Output goes to `data/prepared/<tier>/{train,val,test}/` plus a `manifest.json` with sizes and per-class counts.

> `--source synthetic` generates fake patches. The real reBEN loader is the `TODO` in `real_source()` in `src/prepare_data.py`.

**Training from shards** uses `data=streaming`. A relative `data.data_uri` resolves against the runtime's data root:

| Runtime | `data.data_uri=prepared/dev` resolves to |
|---|---|
| local | `data/prepared/dev` |
| Kaggle | `/kaggle/input/prepared/dev` |
| HF Jobs | `hf://datasets/prepared/dev` |

Absolute paths, `./paths`, `file://...`, `hf://...` and `s3://...` are used as given. `DATA_ROOT=<root>` overrides the root.

```bash
uv run python -m src.train data=streaming data.data_uri=prepared/dev data.image_size=8   # synthetic shards are 8x8
uv run python -m src.train data=streaming data.data_uri=hf://datasets/<user>/reben-dev
```

Shards are read in place: local directories are never copied, and remote ones are fetched chunk by chunk into a cache bounded by `data.max_cache_size`.

## 7. Running on HF Jobs

Requirements: `HF_TOKEN` (write) and `HF_BUCKET` in `.env`, an account that can run Jobs (PRO or an org with billing), and your work **committed and pushed**. The job downloads the code from GitHub at your current commit.

```bash
./scripts/submit_hf.sh --dry-run +experiment=smoke                           # print, don't submit
./scripts/submit_hf.sh --flavor cpu-basic --timeout 30m +experiment=smoke     # cheap end-to-end check
./scripts/submit_hf.sh model=dummy_wide optim.lr=3e-4 trainer.max_epochs=20   # a10g-small, 4h by default
```

The script prints the job id (`<user>/<id>`) and the command to follow it. Use the project's `hf` via `uv run`: another `hf` on your PATH (e.g. from conda) may be too old for these commands.

```bash
uv run hf jobs logs -f <user>/<job_id>     # live JSONL: heartbeats every obs.heartbeat_s seconds
uv run hf jobs inspect <user>/<job_id>     # status, durations, mounts
```

What happens: the job fetches `scripts/hf_entry.py` and the repository tarball at your pinned SHA, runs `uv sync --frozen` (the exact local environment), then `python -m src.train <your overrides>` with `RUN_ENV=hf_jobs`. The bucket is mounted at `/ckpt`, so run directories land in `<bucket>/ckpt/<run_id>/`.

**Get results back:**

```bash
uv run hf buckets sync hf://buckets/$HF_BUCKET/ckpt outputs/hf/ckpt
uv run python -m src.aggregate outputs/hf/ckpt
```

**Evaluate a remote run remotely:**

```bash
./scripts/submit_hf.sh --entry evaluate run_dir=/ckpt/<run_id> degradation=gaussian_noise
```

**Sweeps.** Comma lists expand into the product, one detached job each. Commas inside `[...]` or `{...}` are not split. Job ids (`<user>/<id>`, as `hf jobs` commands accept them) are written to `outputs/sweeps/<timestamp>.tsv`.

```bash
./scripts/sweep_hf.sh --dry-run model=dummy,dummy_wide seed=0,1,2          # 6 commands
./scripts/sweep_hf.sh --flavor t4-small model=dummy,dummy_wide seed=0,1,2
```

Flags for both scripts: `--dry-run`, `--allow-dirty`, `--flavor` (default `a10g-small`), `--timeout` (default `4h`; also passed to the job as `MAX_RUNTIME_S` for the time budget), `--entry train|evaluate`. Use `--` before overrides that start with a dash.

## 8. Running on Kaggle

Requirements: `KAGGLE_API_TOKEN` and `KAGGLE_KERNEL` in `.env`, the Kaggle CLI, and your work committed and pushed.

```bash
./scripts/submit_kaggle.sh --dry-run +experiment=smoke
./scripts/submit_kaggle.sh +experiment=smoke
./scripts/submit_kaggle.sh --kernel <user>/other-slug model=dummy_wide trainer.max_epochs=20
```

The script stamps the commit, `RUN_ENV=kaggle` and your overrides into a copy of `kernels/train/run.py` (Kaggle runs a single code file), then pushes it as a GPU + internet kernel. It polls every 30 s and finally downloads into `outputs/kaggle/<kernel>/<sha>/`: the run directory (`/kaggle/working/<run_id>/`), the Kaggle log, and `kernel_log.json`. The kernel runs in its own `uv sync --frozen` environment rather than on Kaggle's preinstalled packages, which conflict with the pinned versions.

Kaggle sessions have a hard 9 h limit; the time budget (§3) stops cleanly before it. Each push replaces the kernel's previous version, so use a different `--kernel` for parallel runs.

## 9. Reading the logs

`run.jsonl` (also printed to stdout) contains one JSON object per line, each with `ts`, `level`, `event` and `run_id`:

| Event | Meaning |
|---|---|
| `run_start` | Header: entrypoint, runtime, git SHA/dirty, config hash, full config, versions, device, GPU, CPUs, RAM |
| `heartbeat` | Every `obs.heartbeat_s` seconds of **wall time**: phase, epoch, step/max_steps, loss, lr, eta_s, samples/s, data_wait_frac, GPU/host metrics |
| `throughput` | End of each epoch: `samples_per_s`, `data_wait_frac` |
| `data_wait_high` | Warning: more than `obs.warn_data_wait_frac` of the time was spent waiting for data (input-bound) |
| `eval_progress` | One per (degradation, severity) with its metrics |
| `time_budget_stop` | Stopped before the runtime cap, with the resume override |
| `log` | Warnings from Python logging (ours and libraries') |
| `run_end` | Duration, steps, best checkpoint and score (training) or rows written (evaluation) |
| `run_failed` | Exception, traceback, step and config, written before the process exits |

How to read it:
- Heartbeats keep coming while `step` stops changing: the job is **hung**, not just slow.
- `data_wait_frac` near 0 means you are GPU-bound (good). Near 0.3 or above means you are paying for a GPU that waits on I/O.

```bash
grep '"event": "heartbeat"' outputs/<run_id>/run.jsonl | tail -1
python -c "import json,sys; [print(e['event'], e.get('step'), e.get('loss')) for e in map(json.loads, open(sys.argv[1]))]" outputs/<run_id>/run.jsonl
```

## 10. Extending the framework

Every component is one Python class plus one YAML with `_target_`. There is no registry: Hydra instantiates whatever `_target_` names.

**Model** (`src/models/<name>.py` + `configs/model/<name>.yaml`). Subclass `BaseModel`:
- `forward(batch: dict[str, Tensor | None]) -> logits (B, num_classes)`, returning raw logits, never sigmoid.
- Set `required_modalities`.
- A `None` modality must not crash; `tests/test_contracts.py` checks this for every model config automatically.

Take shapes from the data config by interpolation:

```yaml
num_classes: ${data.num_classes}
in_channels: ${data.in_channels}
image_size: ${data.image_size}
```

See the worked `TinyCNN` example in the README.

**Fusion** (`src/models/fusion/<name>.py` + `configs/model/fusion/<name>.yaml`). Subclass `BaseFusion`: `forward(feats: dict[str, Tensor | None]) -> Tensor`, and expose `out_dim`. A model that uses fusion lists it in its YAML defaults (`- fusion: passthrough`). Select it with `model/fusion=<name>`, which is a nested group, so incompatible model/fusion combinations cannot be selected.

**Dataset** (`src/data/<name>.py` + `configs/data/<name>.yaml`). Subclass `BaseDataModule`:
- Batches are dicts `{"<modality>": Tensor[B,C,H,W], ..., "label": Tensor[B,K]}`.
- `num_classes`, `in_channels`, `image_size` must be plain config values, because model configs interpolate them.
- Stream or mount `data_uri`; never download a large archive in `setup()`.

**Degradation** (`src/degradations/<name>.py` + `configs/degradation/<name>.yaml`). Subclass `BaseDegradation`: `__call__(batch, severity, generator) -> batch`. Use **only** the given `generator` for randomness, return the input unchanged at `severity == 0.0`, and don't mutate the input dict. Both properties are tested for every degradation config automatically.

## 11. Development workflow

```bash
uv run pytest -x          # all tests (~3–4 min on Windows; mostly import time)
uv run ruff check src tests kernels
```

Before submitting remotely: commit and push. The scripts refuse a dirty tree because the job would run `HEAD`, not your local edits; `--allow-dirty` overrides this deliberately.

The tests that guard the experiment's validity:
- `tests/test_determinism.py`: identical degraded tensors across training seeds.
- `tests/test_env.py::test_no_module_outside_env_mentions_runtime_specifics`: only `src/env.py` may know which runtime it is on.

## 12. Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `WinError 1114` loading `c10.dll` | venv built on conda's Python (old VC++ runtime). Rebuild with `uv venv --python-preference only-managed`. |
| Every local run takes ~30 s before training starts | Windows Defender scanning torch on import. Harmless; optionally exclude `.venv` and `%LOCALAPPDATA%\uv`. |
| `error: working tree is dirty` | Commit and push first, or pass `--allow-dirty`. |
| `The following secret(s) are not set` | A secret named with `-s` is missing from your environment/`.env`. |
| `.env` value seems wrong | Unquoted values lose a trailing ` # comment`; quote a value that must contain ` #`. |
| `'charmap' codec can't encode` from the Kaggle CLI | Windows console encoding; the script sets `PYTHONUTF8=1`. Set it yourself when calling `kaggle` directly. |
| Kaggle kernel `ERROR` | Read `outputs/kaggle/<kernel>/<sha>/kernel_log.json` or `kaggle kernels logs <kernel>`. |
| `git_sha: "unknown"` in a log | The run had neither a `.git` folder nor `REPO_SHA`. The submission scripts always set it. |
| Permission denied pushing to GitHub | Another GitHub account is stored in Git Credential Manager. Put your username in the remote URL: `https://<user>@github.com/...`. |
| `ClassFloorError` in `prepare_data` | A class has fewer patches than `--min-per-class` in some split. Lower the floor or check the source labels. |

## 13. Reference

**`configs/config.yaml` (training)**

| Key | Default | Meaning |
|---|---|---|
| `seed` | 0 | Training seed (model init, shuffling); seeds dataloader workers too |
| `degradation_seed` | 12345 | Degradation RNG, independent of `seed` |
| `threshold` | 0.5 | Multi-label decision threshold for every metric |
| `output_dir` | null | Durable output root; null = runtime default |
| `results_repo` | null | Hub dataset repo for results rows |
| `resume_from_checkpoint` | null | `.ckpt` to continue from |
| `ckpt_interval_min` | 30 | Wall-clock checkpoint interval |
| `stop_margin_min` | 15 | Stop this long before the runtime cap |
| `obs.heartbeat_s` | 30 | Heartbeat interval (seconds of wall time) |
| `obs.jsonl` | true | Also write `<run dir>/run.jsonl` (stdout JSONL is always on) |
| `obs.warn_data_wait_frac` | 0.3 | Threshold for `data_wait_high` |
| `wandb.enabled` / `project` / `mode` | false / thesis / online | W&B on top of JSONL |

**`configs/evaluate.yaml`:** `run_dir`, `ckpt` (best), `severities` ([0, .25, .5, .75, 1]), `degradation_seed`, `results_repo`, `obs.heartbeat_s`, `obs.jsonl`.

**Config groups:** `model` (dummy, dummy_wide), `model/fusion` (passthrough), `data` (dummy, streaming), `degradation` (none, gaussian_noise), `optim` (adamw), `trainer` (default), `+experiment` (smoke).

**Runtime environment variables** (set by the submission scripts, rarely by hand): `RUN_ENV` (local|kaggle|hf_jobs), `HAS_INTERNET`, `OUTPUT_ROOT`, `DATA_ROOT`, `MAX_RUNTIME_S`, `HYDRA_OVERRIDES` (Kaggle), `REPO_SLUG`, `REPO_SHA`.
