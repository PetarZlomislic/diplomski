"""Experiment 1 pre-flight checks (docs/experiments/exp1-preanalysis.md §8).

Run on the Stage 1 pilot output; Stage 2 must not start unless every check PASSes.

  python -m src.preflight outputs/pilot --ref opt=0.70 --ref sar=0.55

Checks:
  1. identity      severity=0 is bit-identical to the clean batch, for every degradation
                   (on real test batches), and any s=0 result row equals the clean row.
  2. seed          the same degradation_seed gives byte-identical corrupted tensors when
                   the test data is loaded through two different trained runs' configs.
  3. leakage       `opt` is unaffected by SAR degradations and `sar` by optical ones
                   (results must equal the clean row exactly).
  4. reference     clean macro-F1 of opt / sar within --tolerance of the published
                   BIFOLD S2-only / S1-only numbers given with --ref (SKIP without them).
  5. data_wait     median training data_wait_frac < 0.3 for every run.
  6. per_class     no class has F1 exactly 0 in every config (clean test rows).
Exit status 0 only if every check PASSes.
"""
import argparse
import json
import statistics
import sys
from pathlib import Path

import pandas as pd
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

from src.aggregate import load_results
from src.evaluate import degraded_batches
from src.results import RUN_CONFIG

DEGRADATIONS = ("cloud", "speckle", "drop_optical", "drop_sar", "temporal")
MAX_SEVERITY = {"cloud": 1.0, "speckle": 1.0, "drop_optical": 1.0, "drop_sar": 1.0,
                "temporal": 1}
# config -> degradations that must not move it (they act on a modality it does not use)
MUST_NOT_MOVE = {"opt": ("speckle", "drop_sar", "temporal"),
                 "sar": ("cloud", "drop_optical")}
CONFIG_DIR = Path(__file__).resolve().parent.parent / "configs"


def _same(a: dict, b: dict) -> bool:
    if a.keys() != b.keys():
        return False
    for k in a:
        x, y = a[k], b[k]
        if isinstance(x, torch.Tensor) or isinstance(y, torch.Tensor):
            if not (isinstance(x, torch.Tensor) and isinstance(y, torch.Tensor)
                    and x.dtype == y.dtype and torch.equal(x, y)):
                return False
        elif x != y:
            return False
    return True


def _degradation(name: str):
    cfg = OmegaConf.load(CONFIG_DIR / "degradation" / f"{name}.yaml")
    return instantiate(cfg)


def _datamodule(run_dir: Path):
    dm = instantiate(OmegaConf.load(run_dir / RUN_CONFIG).cfg.data)
    dm.setup("test")
    return dm


def _first(it, n: int) -> list:
    out = []
    for i, x in enumerate(it):
        if i >= n:
            break
        out.append(x)
    return out


def check_identity(dm, degradations, n_batches: int, df: pd.DataFrame) -> tuple[str, str]:
    bad = []
    for name in degradations:
        deg = _degradation(name)
        deg.bind(dm)
        clean = [dm.prepare_batch(b) for b in _first(iter(dm.test_dataloader()), n_batches)]
        zero = _first(degraded_batches(dm, deg, 0.0, 12345), n_batches)
        if not all(_same(a, b) for a, b in zip(clean, zero, strict=True)):
            bad.append(f"{name}: tensors differ at s=0")
    clean = df[df["degradation"] == "none"].set_index("run_id")["f1_macro"]
    zero_rows = df[(df["degradation"] != "none") & (df["severity"] == 0.0)]
    for _, r in zero_rows.iterrows():
        if r["run_id"] in clean and r["f1_macro"] != clean[r["run_id"]]:
            bad.append(f"{r['run_id']} {r['degradation']}: s=0 row != clean row")
    return ("FAIL", "; ".join(bad)) if bad else (
        "PASS", f"{len(degradations)} degradations x {n_batches} batches; "
                f"{len(zero_rows)} s=0 result rows")


def check_seed(run_dirs: list[Path], degradations, n_batches: int,
               degradation_seed: int) -> tuple[str, str]:
    if len(run_dirs) < 2:
        return "SKIP", "needs two runs"
    a_dir, b_dir = run_dirs[0], next(
        (d for d in run_dirs[1:] if _model(d) != _model(run_dirs[0])), run_dirs[1])
    dms = [_datamodule(a_dir), _datamodule(b_dir)]
    bad = []
    for name in degradations:
        outs = []
        for dm in dms:
            deg = _degradation(name)
            deg.bind(dm)
            outs.append(_first(degraded_batches(dm, deg, MAX_SEVERITY[name],
                                                degradation_seed), n_batches))
        if not all(_same(x, y) for x, y in zip(*outs, strict=True)):
            bad.append(name)
    detail = f"{_model(a_dir)} vs {_model(b_dir)}"
    return ("FAIL", f"{detail}: differ under {bad}") if bad else ("PASS", detail)


def _model(run_dir: Path) -> str:
    return str(OmegaConf.load(run_dir / RUN_CONFIG).meta.model)


def check_leakage(df: pd.DataFrame) -> tuple[str, str]:
    clean = df[df["degradation"] == "none"].set_index("run_id")["f1_macro"]
    bad, checked = [], 0
    for model, degs in MUST_NOT_MOVE.items():
        rows = df[(df["model"] == model) & df["degradation"].isin(degs)]
        for _, r in rows.iterrows():
            checked += 1
            if r["f1_macro"] != clean.get(r["run_id"]):
                bad.append(f"{model}/{r['degradation']}@{r['severity']:g}: "
                           f"{r['f1_macro']:.4f} != clean {clean.get(r['run_id'])}")
    if not checked:
        return "SKIP", "no opt/sar rows under the other modality's degradations"
    return ("FAIL", "degradation leaks into the wrong modality: " + "; ".join(bad)) if bad \
        else ("PASS", f"{checked} rows equal clean")


def check_reference(df: pd.DataFrame, refs: dict[str, float], tol: float) -> tuple[str, str]:
    if not refs:
        return "SKIP", "no --ref given (published BIFOLD S2-only / S1-only macro-F1)"
    clean = df[df["degradation"] == "none"]
    bad, ok = [], []
    for model, ref in refs.items():
        v = clean[clean["model"] == model]["f1_macro"]
        if v.empty:
            bad.append(f"{model}: no clean rows")
            continue
        (ok if abs(v.mean() - ref) <= tol else bad).append(
            f"{model} {v.mean():.3f} vs ref {ref:.3f}")
    return ("FAIL", "; ".join(bad)) if bad else ("PASS", "; ".join(ok))


def check_data_wait(run_dirs: list[Path], limit: float) -> tuple[str, str]:
    worst, bad, seen = {}, [], 0
    for d in run_dirs:
        log = d / "run.jsonl"
        if not log.exists():
            continue
        vals = []
        for line in log.read_text().splitlines():
            e = json.loads(line)
            if e.get("event") == "heartbeat" and e.get("data_wait_frac") is not None:
                vals.append(float(e["data_wait_frac"]))
        if vals:
            seen += 1
            worst[d.name] = statistics.median(vals)
            if worst[d.name] >= limit:
                bad.append(f"{d.name} ({_model(d)}) {worst[d.name]:.2f}")
    if not seen:
        return "SKIP", "no heartbeat with data_wait_frac in any run.jsonl"
    return ("FAIL", f">= {limit}: " + "; ".join(bad)) if bad else (
        "PASS", f"max median {max(worst.values()):.2f} over {seen} runs")


def check_per_class(df: pd.DataFrame) -> tuple[str, str]:
    clean = df[(df["degradation"] == "none") & df["f1_per_class"].notna()]
    if clean.empty:
        return "SKIP", "no per-class F1 in clean rows"
    per_model = clean.groupby("model")["f1_per_class"].apply(
        lambda s: [max(v) for v in zip(*(json.loads(x) for x in s))])
    n = len(per_model.iloc[0])
    dead = [k for k in range(n) if all(v[k] == 0.0 for v in per_model)]
    return ("FAIL", f"classes with F1 = 0 in every config: {dead}") if dead else (
        "PASS", f"{n} classes, {len(per_model)} configs")


def preflight(root: Path, refs: dict[str, float], tol: float, n_batches: int,
              degradations=DEGRADATIONS, degradation_seed: int = 12345,
              data_wait_limit: float = 0.3) -> dict[str, tuple[str, str]]:
    run_dirs = sorted(p.parent for p in root.rglob(RUN_CONFIG))
    if not run_dirs:
        raise FileNotFoundError(f"no {RUN_CONFIG} under {root}")
    df = load_results(root)
    dm = _datamodule(run_dirs[0])
    return {
        "1 identity": check_identity(dm, degradations, n_batches, df),
        "2 seed": check_seed(run_dirs, degradations, n_batches, degradation_seed),
        "3 leakage": check_leakage(df),
        "4 reference": check_reference(df, refs, tol),
        "5 data_wait": check_data_wait(run_dirs, data_wait_limit),
        "6 per_class": check_per_class(df),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", type=Path, help="directory containing the pilot run dirs")
    ap.add_argument("--ref", action="append", default=[], metavar="MODEL=F1",
                    help="published clean macro-F1, e.g. opt=0.70 (repeatable)")
    ap.add_argument("--tolerance", type=float, default=0.05)
    ap.add_argument("--batches", type=int, default=5, help="test batches per tensor check")
    ap.add_argument("--degradations", nargs="*", default=list(DEGRADATIONS))
    args = ap.parse_args()
    refs = {k: float(v) for k, v in (r.split("=", 1) for r in args.ref)}
    results = preflight(args.root, refs, args.tolerance, args.batches, args.degradations)
    for name, (status, detail) in results.items():
        print(f"[{status:4}] {name:12} {detail}")
    ready = all(s == "PASS" for s, _ in results.values())
    print("READY for Stage 2" if ready else "NOT READY: every check must PASS (§8)")
    sys.exit(0 if ready else 1)


if __name__ == "__main__":
    main()
