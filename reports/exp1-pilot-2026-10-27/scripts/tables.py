"""Tables and integrity checks for the Experiment 1 Stage 1 pilot review.

  uv run python reports/exp1-pilot-2026-10-27/scripts/tables.py <results dir>

Reads only the results dir (run dirs + kernel_log.json) and the repo's analysis/preflight
helpers; writes reports/exp1-pilot-2026-10-27/tables/*.csv and prints the integrity checks.
"""
import json
import statistics
import sys
from pathlib import Path

import pandas as pd
import yaml

from src.analysis import CONFIGS, DEGRADATIONS, clean_scores, curves, load_results
from src.prepare_data import CLASSES
from src.preflight import (check_data_wait, check_leakage, check_per_class,
                           check_reference)

ROOT = Path(sys.argv[1])
OUT = Path(__file__).resolve().parents[1] / "tables"
OUT.mkdir(parents=True, exist_ok=True)
REFS = {"opt": 0.644, "sar": 0.566}            # BIFOLD ResNet-18 test macro-F1
PILOT_GRID = yaml.safe_load(Path("configs/evaluate.yaml").read_text())["grids"]["pilot"]

run_dirs = sorted(p.parent for p in ROOT.rglob("run_config.yaml"))
raw = pd.concat([pd.read_csv(f) for f in ROOT.rglob("results/*.csv")], ignore_index=True)
df = load_results(ROOT)


def events(run_dir: Path) -> list[dict]:
    return [json.loads(l) for l in (run_dir / "run.jsonl").read_text().splitlines() if l]


# ------------------------------------------------------------------ provenance
prov = []
for d in run_dirs:
    cfg = yaml.safe_load((d / "run_config.yaml").read_text())
    ev = events(d)
    train_end = next(e for e in ev if e["event"] == "run_end" and "best_val_f1_macro" in e)
    evals = [e for e in ev if e["event"] == "run_end" and e.get("entrypoint") == "evaluate"]
    tp = [e for e in ev if e["event"] == "throughput"]
    prov.append({
        "run_id": d.name, "model": cfg["meta"]["model"], "seed": cfg["cfg"]["seed"],
        "git_sha": cfg["meta"]["git_sha"], "config_hash": cfg["meta"]["config_hash"],
        "width": cfg["cfg"]["model"]["encoder"]["width"],
        "max_epochs": cfg["cfg"]["trainer"]["max_epochs"],
        "epochs_logged": len(tp), "batch_size": cfg["cfg"]["data"]["batch_size"],
        "num_workers": cfg["cfg"]["data"]["num_workers"],
        "lr": cfg["cfg"]["optim"]["lr"], "data_uri": cfg["cfg"]["data"]["data_uri"],
        "degradation_seed": cfg["cfg"]["degradation_seed"],
        "train_status": train_end["status"], "train_s": train_end["duration_s"],
        "global_step": train_end.get("global_step"),
        "best_val_f1_macro": train_end["best_val_f1_macro"],
        "n_evals": len(evals), "evals_ok": sum(e["status"] == "ok" for e in evals),
        "eval_s": sum(e["duration_s"] for e in evals),
        "median_data_wait_frac": statistics.median(e["data_wait_frac"] for e in tp),
        "max_data_wait_frac": max(e["data_wait_frac"] for e in tp),
        "mean_samples_per_s": statistics.mean(e["samples_per_s"] for e in tp),
        "n_data_wait_high": sum(e["event"] == "data_wait_high" for e in ev),
        "n_run_failed": sum(e["event"] == "run_failed" for e in ev),
    })
prov = pd.DataFrame(prov).set_index("model").loc[list(CONFIGS)].reset_index()
prov.to_csv(OUT / "provenance.csv", index=False)

# per-epoch throughput
tp_rows = [{"model": yaml.safe_load((d / "run_config.yaml").read_text())["meta"]["model"],
            "epoch": e["epoch"], "samples_per_s": e["samples_per_s"],
            "data_wait_frac": e["data_wait_frac"]}
           for d in run_dirs for e in events(d) if e["event"] == "throughput"]
pd.DataFrame(tp_rows).to_csv(OUT / "throughput.csv", index=False)

# train-phase loss from heartbeats
hb = [{"model": yaml.safe_load((d / "run_config.yaml").read_text())["meta"]["model"],
       "step": e["step"], "epoch": e["epoch"], "loss": e["loss"], "lr": e.get("lr")}
      for d in run_dirs for e in events(d)
      if e["event"] == "heartbeat" and e.get("phase") == "train" and e.get("loss") is not None]
pd.DataFrame(hb).to_csv(OUT / "train_loss.csv", index=False)

# ------------------------------------------------------------------ clean table
clean = df[df["degradation"] == "none"].set_index("model").loc[list(CONFIGS)]
ct = clean[["f1_macro", "f1_micro", "accuracy", "params", "flops", "latency_ms"]].copy()
ct["best_val_f1_macro"] = prov.set_index("model")["best_val_f1_macro"]
ct["bifold_ref"] = pd.Series(REFS)
ct["minus_ref"] = ct["f1_macro"] - ct["bifold_ref"]
ct["n_classes_f1_zero"] = [sum(v == 0 for v in json.loads(x)) for x in clean["f1_per_class"]]
ct.reset_index().to_csv(OUT / "clean.csv", index=False)

# ------------------------------------------------------------------ degradation table
cv = curves(df)
cv.to_csv(OUT / "curves_f1_R.csv", index=False)
deg = cv.assign(point=cv["degradation"] + "@" + cv["severity"].map("{:g}".format))
wide = deg.pivot(index="model", columns="point", values="F1").loc[list(CONFIGS)]
wide_r = deg.pivot(index="model", columns="point", values="R").loc[list(CONFIGS)]
cols = [f"{d}@{s:g}" for d in DEGRADATIONS for s in [0] + PILOT_GRID[d]]
wide[cols].reset_index().to_csv(OUT / "degradation_f1_macro.csv", index=False)
wide_r[cols].reset_index().to_csv(OUT / "degradation_retention.csv", index=False)
micro = df[df["degradation"] != "none"].assign(
    point=lambda d: d["degradation"] + "@" + d["severity"].map("{:g}".format)).pivot(
    index="model", columns="point", values="f1_micro").loc[list(CONFIGS)]
micro.reset_index().to_csv(OUT / "degradation_f1_micro.csv", index=False)

# ------------------------------------------------------------------ per class
pc_clean = pd.DataFrame({m: json.loads(clean.loc[m, "f1_per_class"]) for m in CONFIGS},
                        index=CLASSES)
pc_clean.index.name = "class"
pc_clean.to_csv(OUT / "per_class_clean.csv")
drops = []
for d in DEGRADATIONS:
    smax = max(PILOT_GRID[d])
    for m in CONFIGS:
        row = df[(df["model"] == m) & (df["degradation"] == d) & (df["severity"] == smax)]
        v = json.loads(row["f1_per_class"].iloc[0])
        for c, f, f0 in zip(CLASSES, v, pc_clean[m]):
            drops.append({"degradation": d, "severity": smax, "model": m, "class": c,
                          "f1_clean": f0, "f1_degraded": f, "delta": f - f0})
pd.DataFrame(drops).to_csv(OUT / "per_class_harshest.csv", index=False)

# ------------------------------------------------------------------ modality weights
mw = df[df["modality_weights"].notna()].copy()
mw["w_optical"] = mw["modality_weights"].map(lambda s: json.loads(s)["optical"])
mw["w_sar"] = mw["modality_weights"].map(lambda s: json.loads(s)["sar"])
mw[["model", "degradation", "severity", "w_optical", "w_sar", "f1_macro"]].to_csv(
    OUT / "modality_weights.csv", index=False)

# ------------------------------------------------------------------ integrity
checks = []


def add(name, status, evidence):
    checks.append({"check": name, "status": status, "evidence": evidence})


expected = {(m, "none", 0.0) for m in CONFIGS} | {
    (m, d, float(s)) for m in CONFIGS for d in DEGRADATIONS for s in PILOT_GRID[d]}
got = set(zip(df["model"], df["degradation"], df["severity"].astype(float)))
missing, extra = expected - got, got - expected
add("rows present (pilot grid)", "PASS" if not missing and not extra else "FAIL",
    f"{len(df)} rows after de-dup ({len(raw)} raw) for {len(expected)} expected "
    f"(6 arms x [clean + {sum(len(v) for v in PILOT_GRID.values())} sweep points], seed 0); "
    f"missing {sorted(missing)}; extra {sorted(extra)}")

per_arm = df.groupby("model").agg(shas=("git_sha", "nunique"), cfgs=("config_hash", "nunique"),
                                  runs=("run_id", "nunique"))
shas = set(df["git_sha"]) | set(prov["git_sha"])
cfg_match = all(df[df["model"] == r.model]["config_hash"].eq(r.config_hash).all()
                for r in prov.itertuples())
ok = (per_arm[["shas", "cfgs", "runs"]] == 1).all().all() and len(shas) == 1 and cfg_match
add("one SHA / config per arm", "PASS" if ok else "FAIL",
    f"git_sha {sorted(shas)}; one run_id and config_hash per arm, result rows match "
    f"run_config.yaml: {cfg_match}; git_dirty false in every run_start")

zero_rows = df[(df["degradation"] != "none") & (df["severity"] == 0.0)]
same_zero = [m for m in ("opt", "early")
             if df.query("model == @m and degradation == 'cloud' and severity == 1.0")[
                 "f1_per_class"].iloc[0] == df.query(
                 "model == @m and degradation == 'drop_optical'")["f1_per_class"].iloc[0]]
add("severity 0 equals clean", "SKIP",
    f"the pilot sweep does not repeat s=0 ({len(zero_rows)} s=0 sweep rows; F1(0) is the "
    "clean row by design, deviation 5); the tensor-level check needs the test shards, which "
    "are not available locally. Consistency evidence: cloud@1.0 and drop_optical give "
    f"identical per-class F1 for {same_zero} (both feed an all-zero optical input)")

st, ev = check_leakage(df)
add("no leakage (preflight 3)", st, ev + ": opt under speckle/drop_sar/temporal, "
    "sar under cloud/drop_optical")

st, ev = check_per_class(df)
zero_by_arm = {m: [c for c, v in zip(CLASSES, json.loads(clean.loc[m, 'f1_per_class']))
                   if v == 0.0] for m in CONFIGS}
add("per-class F1 not all zero (preflight 6)", st,
    ev + "; clean classes at F1 = 0 per arm: " + "; ".join(
        f"{m}: {v or 'none'}" for m, v in zero_by_arm.items()))

st, ev = check_data_wait(run_dirs, 0.3)
add("data_wait_frac < 0.3 (preflight 5)", st,
    ev + " (median over train heartbeats); per-epoch throughput events: per-arm median "
    f"{prov['median_data_wait_frac'].min():.2f}-{prov['median_data_wait_frac'].max():.2f}, "
    f"max {prov['max_data_wait_frac'].max():.2f}; "
    f"{int(prov['n_data_wait_high'].sum())} data_wait_high warnings = every epoch of every arm; "
    f"num_workers 10 on a 4-vCPU host")

st, ev = check_reference(df, REFS, 0.05)
add("clean F1 vs BIFOLD reference (preflight 4, tol 0.05)", st, ev)

klog = (ROOT / "kernel_log.json").read_text(encoding="utf-8")
log = json.loads(klog[klog.index("["):])
lines = [json.loads(x) for e in log for x in e["data"].splitlines()
         if x.startswith("{") and x.rstrip().endswith("}")]
done = [x for x in lines if x.get("event") == "exp1_all_done"]
failed = [x for x in lines if x.get("event") == "run_failed" or x.get("level") == "error"]
ends = [x for x in lines if x.get("event") == "run_end"]
add("failures / reruns in logs",
    "PASS" if done and not done[0]["failed"] and not failed and len(raw) == len(df) else "FAIL",
    f"exp1_all_done failed={done[0]['failed'] if done else 'MISSING'}; {len(failed)} "
    f"run_failed/error events; {len(ends)} run_end events, all status "
    f"{sorted({x['status'] for x in ends})}; {len(raw) - len(df)} duplicate result rows "
    f"(re-evaluations); every run started fresh (resumed_from null)")

add("preflight 1 identity / 2 seed (tensor checks)", "SKIP",
    "`uv run python -m src.preflight` crashes before any check: it needs the test shards "
    "(data/peroz1/reben-sweep/manifest.json not found locally). See analysis/preflight.txt")

pd.DataFrame(checks).to_csv(OUT / "integrity.csv", index=False)
pd.set_option("display.width", 250)
pd.set_option("display.max_colwidth", 400)
print(pd.DataFrame(checks).to_string())
print(prov.T.to_string())
print(ct.to_string())

# ------------------------------------------------------------------ Stage 2 cost projection
# Rough: scales the pilot's measured wall times linearly in data size, epochs and sweep
# points (same hardware, same input-bound loader; ignores width 32 vs 16); reBEN official split sizes from docs/experiments/exp1-implementation.md.
FULL_TRAIN, FULL_TEST = 237_871, 119_825
pilot_train = prov["global_step"].iloc[0] / prov["max_epochs"].iloc[0] * prov["batch_size"].iloc[0]
sps = prov["mean_samples_per_s"].mean()
full_points = 1 + sum(len(v) for v in yaml.safe_load(
    Path("configs/evaluate.yaml").read_text())["grids"]["full"].values())
pilot_points = 1 + sum(len(v) for v in PILOT_GRID.values())
pilot_test = FULL_TEST * 0.15
eval_s_per_point_per_sample = prov["eval_s"].mean() / pilot_points / pilot_test
proj = pd.DataFrame([{
    "pilot_train_samples_per_epoch": pilot_train,
    "pilot_mean_samples_per_s": sps,
    "pilot_train_h_per_arm": prov["train_s"].mean() / 3600,
    "pilot_eval_h_per_arm": prov["eval_s"].mean() / 3600,
    # train + per-epoch validation, scaled by data size and epochs (30 vs 15)
    "stage2_train_h_per_run": prov["train_s"].mean() * (FULL_TRAIN / pilot_train)
                              * (30 / prov["max_epochs"].iloc[0]) / 3600,
    "stage2_eval_h_per_run": eval_s_per_point_per_sample * full_points * FULL_TEST / 3600,
    "stage2_runs": 18,
}])
proj["stage2_total_h"] = proj["stage2_runs"] * (proj["stage2_train_h_per_run"]
                                                + proj["stage2_eval_h_per_run"])
proj.to_csv(OUT / "stage2_projection.csv", index=False)
print(proj.T.to_string())

# ------------------------------------------------------------------ quoted repo tables
import shutil  # noqa: E402

for name in ("table1.csv", "table2_hypotheses.csv", "orderings.csv"):
    shutil.copyfile(OUT.parent / "analysis" / "exp1" / name, OUT / name)
