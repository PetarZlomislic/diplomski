"""Experiment 1, one process: train each (model, seed), then sweep it under the stage grid.

  python -m src.exp1_run models=[opt,sar,early,late,gate_global,gate_spatial] seeds=[0] \
      stage=pilot "train=[+experiment=exp1_pilot,data.data_uri=peroz1/reben-sweep]"

Made for runtimes where one long session is cheaper than many jobs (the time budget in
src.train still applies to every run). Arguments are key=value pairs:
  models  list of model config names          seeds  list of training seeds
  stage   sweep grid in configs/evaluate.yaml (pilot|full|none)
  train   extra Hydra overrides for every training run
A run that fails is logged and the next one starts; the exit status is non-zero if any did.
"""
import json
import sys
import time
import traceback

from omegaconf import OmegaConf

from src.env import detect


def _event(event: str, **fields) -> None:
    print(json.dumps({"ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                      "event": event, **fields}), flush=True)


def main(argv: list[str]) -> int:
    from hydra import compose, initialize_config_dir

    from src.evaluate import CONFIG_DIR, sweep
    from src.train import run

    args = OmegaConf.from_dotlist(argv)
    models = list(args.get("models", []))
    seeds = list(args.get("seeds", [0]))
    stage = str(args.get("stage", "pilot"))
    extra = [str(o) for o in args.get("train", [])]
    if not models:
        raise SystemExit("models=[...] is required")
    failed = []
    for seed in seeds:
        for model in models:
            label = f"{model} seed={seed}"
            _event("exp1_run_start", model=model, seed=seed, stage=stage)
            try:
                with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base="1.3"):
                    cfg = compose("config", overrides=[*extra, f"model={model}", f"seed={seed}"])
                    run_dir = run(cfg, {"model": model, "data": "streaming"})
                    if stage != "none":
                        sweep(compose("evaluate", overrides=[f"run_dir={run_dir.as_posix()}",
                                                             f"stage={stage}"]))
                _event("exp1_run_done", model=model, seed=seed, run_dir=str(run_dir))
            except Exception as e:  # keep going: one broken arm must not cost the session
                traceback.print_exc()
                failed.append(label)
                _event("exp1_run_failed", model=model, seed=seed, exception=repr(e)[:300])
    _event("exp1_all_done", failed=failed)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(detect().hydra_argv(sys.argv)[1:]))
