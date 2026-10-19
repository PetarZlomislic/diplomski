"""Results analysis.

  python -m src.analysis aggregate outputs   # merged CSV + one degradation-curve plot each
  python -m src.analysis exp1 outputs        # Experiment 1 tables, verdicts and figures

`aggregate`: every <run>/results/*.csv merged into results_all.csv, plus a curve per
degradation (mean over seeds, min-max band).

`exp1` (docs/experiments/exp1-preanalysis.md §3-§5, §9): results dir -> Table 1 (clean,
AUDC, AUDC_rel, cost), Table 2 (hypothesis verdicts), ordering disagreements, Figures 1-3.

Conventions, fixed before any result (see the plan's deviations log):
  * F1(0) for every degradation is the run's clean `none` row; a sweep need not include 0.
  * R(s) = F1(s) / F1(0) per run; AUDC = mean_s F1(s), AUDC_rel = mean_s R(s), over the
    grid INCLUDING s = 0, computed per run and then summarised over seeds.
  * "Exceeds the seed spread" = the two configs' seed ranges do not overlap
    (min of one > max of the other). With fewer than 2 seeds a verdict is "inconclusive".
  * H5 "correlates" = |Spearman rho| >= 0.8 between params and mean AUDC across configs.
"""
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.evaluate import FIELDNAMES, RESULTS_DIR  # noqa: E402

CONFIGS = ("opt", "sar", "early", "late", "gate_global", "gate_spatial")
FUSION = ("early", "late", "gate_global", "gate_spatial")
GATES = ("gate_global", "gate_spatial")
DEGRADATIONS = ("cloud", "speckle", "drop_optical", "drop_sar", "temporal")
OPTICAL_DEGRADATIONS = ("cloud", "drop_optical")
H5_RHO = 0.8
METRIC = "f1_macro"
METRICS = ["f1_macro", "f1_micro", "accuracy"]
# Validated categorical palette (light surface), assigned in fixed order, never cycled.
SERIES_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
                 "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
SURFACE, INK, INK_MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"


def load_results(root: Path) -> pd.DataFrame:
    files = sorted(p for p in root.rglob("*.csv") if p.parent.name == RESULTS_DIR)
    if not files:
        raise FileNotFoundError(f"no results/*.csv under {root}")
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df = df.reindex(columns=FIELDNAMES)  # files from before a column was added: NaN
    # Re-evaluating the same checkpoint supersedes earlier rows instead of double-counting.
    key = ["run_id", "degradation", "severity", "ckpt_kind", "threshold"]
    df = df.sort_values("timestamp").drop_duplicates(key, keep="last")
    return df.sort_values(["model", "degradation", "severity"], ignore_index=True)


def model_colors(models: list[str]) -> dict[str, str]:
    """Color follows the model name (sorted), not its rank in a given figure."""
    names = sorted(set(models))
    if len(names) > len(SERIES_COLORS):
        raise ValueError(f"{len(names)} models > {len(SERIES_COLORS)} colors: facet instead")
    return dict(zip(names, SERIES_COLORS))


# ---------------------------------------------------------------- per-run quantities

def clean_scores(df: pd.DataFrame, metric: str = METRIC) -> pd.Series:
    """run_id -> clean test metric (the `none` row)."""
    clean = df[(df["degradation"] == "none") & (df["severity"] == 0.0)]
    return clean.drop_duplicates("run_id", keep="last").set_index("run_id")[metric]


def curves(df: pd.DataFrame, metric: str = METRIC) -> pd.DataFrame:
    """One row per (run, degradation, severity) with F1 and R, the clean point inserted at
    s = 0 for every degradation a run was swept on."""
    clean = clean_scores(df, metric)
    meta = df.drop_duplicates("run_id", keep="last").set_index("run_id")[["model", "seed"]]
    swept = df[df["degradation"] != "none"][["run_id", "degradation", "severity", metric]]
    zero = (swept[["run_id", "degradation"]].drop_duplicates()
            .assign(severity=0.0, **{metric: lambda d: d["run_id"].map(clean)}))
    have_zero = swept[swept["severity"] == 0.0][["run_id", "degradation"]]
    zero = zero.merge(have_zero, how="left", indicator=True)
    zero = zero[zero["_merge"] == "left_only"].drop(columns="_merge")
    out = pd.concat([swept, zero], ignore_index=True).dropna(subset=[metric])
    out = out.join(meta, on="run_id")
    out["clean"] = out["run_id"].map(clean)
    out["R"] = out[metric] / out["clean"]
    return out.rename(columns={metric: "F1"}).sort_values(
        ["degradation", "model", "seed", "severity"], ignore_index=True)


def per_run(df: pd.DataFrame, cv: pd.DataFrame, metric: str = METRIC) -> pd.DataFrame:
    """(run, degradation) -> clean, AUDC, AUDC_rel, n_severities. Every run also gets a
    degradation="none" row (AUDC = clean, AUDC_rel = 1), so clean-only runs count too."""
    g = cv.groupby(["run_id", "model", "seed", "degradation"], as_index=False)
    swept = g.agg(clean=("clean", "first"), AUDC=("F1", "mean"), AUDC_rel=("R", "mean"),
                  n_severities=("severity", "nunique"))
    meta = df.drop_duplicates("run_id", keep="last").set_index("run_id")[["model", "seed"]]
    clean = clean_scores(df, metric).rename("clean").to_frame().join(meta).reset_index()
    clean = clean.assign(degradation="none", AUDC=clean["clean"], AUDC_rel=1.0,
                         n_severities=1)
    return pd.concat([clean[swept.columns], swept], ignore_index=True)


# ---------------------------------------------------------------- Table 1

def _band(s: pd.Series) -> dict[str, float]:
    return {"mean": s.mean(), "min": s.min(), "max": s.max()}


def table1(df: pd.DataFrame, runs: pd.DataFrame) -> pd.DataFrame:
    clean_rows = df[(df["degradation"] == "none") & (df["severity"] == 0.0)]
    cost = clean_rows.groupby("model").agg(params=("params", "first"),
                                           flops=("flops", "mean"),
                                           latency_ms=("latency_ms", "mean"))
    rows = []
    for (model, deg), g in runs.groupby(["model", "degradation"]):
        row = {"model": model, "degradation": deg, "n_seeds": g["seed"].nunique(),
               "n_severities": int(g["n_severities"].max())}
        for q in ("clean", "AUDC", "AUDC_rel"):
            row.update({f"{q}_{k}": v for k, v in _band(g[q]).items()})
        rows.append(row)
    t = pd.DataFrame(rows).join(cost, on="model")
    for col in ("AUDC_mean", "AUDC_rel_mean"):
        # Nullable: AUDC_rel is undefined (NaN) for a config whose clean F1 is 0, and an
        # undefined score gets no rank instead of crashing the analysis.
        t[f"rank_{col.removesuffix('_mean')}"] = (
            t.groupby("degradation")[col].rank(ascending=False, method="min").astype("Int64"))
    return t.sort_values(["degradation", "rank_AUDC_rel"], ignore_index=True)


def orderings(t1: pd.DataFrame) -> pd.DataFrame:
    """Per degradation: config order by absolute AUDC vs by AUDC_rel. A disagreement is a
    reported finding (§3), not something to resolve."""
    rows = []
    for deg, g in t1.groupby("degradation"):
        a = g.sort_values("AUDC_mean", ascending=False)["model"].tolist()
        r = g.sort_values("AUDC_rel_mean", ascending=False)["model"].tolist()
        rows.append({"degradation": deg, "order_AUDC": " > ".join(a),
                     "order_AUDC_rel": " > ".join(r), "disagree": a != r})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- Table 2

def _vals(runs: pd.DataFrame, model: str, deg: str | None, col: str) -> np.ndarray:
    g = runs[runs["model"] == model]
    if deg is not None:
        g = g[g["degradation"] == deg]
    else:  # clean: one value per run
        g = g.drop_duplicates("run_id")
    return g[col].to_numpy(dtype=float)


def beats(a: np.ndarray, b: np.ndarray) -> bool | None:
    """a exceeds b by more than the seed spread (ranges disjoint); None if undecidable."""
    if len(a) < 2 or len(b) < 2:
        return None
    return bool(a.min() > b.max())


def _fmt(name: str, v: np.ndarray) -> str:
    if len(v) == 0:
        return f"{name} n/a"
    if len(v) == 1:
        return f"{name} {v[0]:.3f}"
    return f"{name} {v.mean():.3f} [{v.min():.3f}-{v.max():.3f}]"


def _verdict(held: bool, rejected: bool, undecided: bool) -> str:
    if rejected:
        return "rejected"
    if undecided:
        return "inconclusive"
    return "held" if held else "inconclusive"


def hypotheses(runs: pd.DataFrame) -> pd.DataFrame:
    models = set(runs["model"])
    degs = set(runs["degradation"])
    out = []

    def add(h, scope, verdict, evidence, action=""):
        out.append({"hypothesis": h, "scope": scope, "verdict": verdict,
                    "evidence": evidence, "action": action})

    # H1 optical dominance on clean data.
    fusion = [f for f in FUSION if f in models]
    if "opt" in models and fusion:
        opt = _vals(runs, "opt", None, "clean")
        b = {f: beats(_vals(runs, f, None, "clean"), opt) for f in fusion}
        verdict = _verdict(held=all(v is False for v in b.values()),
                           rejected=any(v is True for v in b.values()),
                           undecided=any(v is None for v in b.values()))
        ev = "; ".join([_fmt("opt", opt)] + [_fmt(f, _vals(runs, f, None, "clean"))
                                             for f in fusion])
        add("H1", "clean macro-F1", verdict, ev,
            "expected; fusion value must show under degradation" if verdict == "held" else "")

    # H2 asymmetric degradation: drop under cloud > drop under speckle, per fusion config.
    if {"cloud", "speckle"} <= degs and fusion:
        held, rejected, undecided, ev = True, False, False, []
        for f in fusion:
            c = runs[(runs["model"] == f) & (runs["degradation"] == "cloud")].set_index("run_id")
            s = runs[(runs["model"] == f) & (runs["degradation"] == "speckle")].set_index("run_id")
            both = c.index.intersection(s.index)
            diff = ((c.loc[both, "clean"] - c.loc[both, "AUDC"])
                    - (s.loc[both, "clean"] - s.loc[both, "AUDC"])).to_numpy(dtype=float)
            if len(diff) == 0:
                continue
            ev.append(_fmt(f"{f} drop(cloud)-drop(speckle)", diff))
            rejected |= diff.mean() <= 0
            undecided |= len(diff) < 2
            held &= len(diff) >= 2 and diff.min() > 0
        add("H2", "AUDC drop, cloud vs speckle", _verdict(held, rejected, undecided),
            "; ".join(ev))

    # H3 graceful-degradation ordering under optical degradation, on AUDC_rel.
    if set(GATES) | {"late", "early"} <= models:
        for deg in [d for d in OPTICAL_DEGRADATIONS if d in degs]:
            v = {m: _vals(runs, m, deg, "AUDC_rel") for m in (*GATES, "late", "early")}
            b = [beats(v[g], v["late"]) for g in GATES] + [beats(v["late"], v["early"])]
            rejected = any(v["early"].mean() >= v[g].mean() for g in GATES)
            verdict = _verdict(all(x is True for x in b), rejected, any(x is None for x in b))
            add("H3", f"AUDC_rel under {deg}", verdict,
                "; ".join(_fmt(m, x) for m, x in v.items()),
                "gate not learning what is claimed" if rejected else "")

    # H4 granularity x structure: spatial beats global under cloud only.
    if set(GATES) <= models and "cloud" in degs:
        others = [d for d in ("speckle", "drop_optical", "drop_sar") if d in degs]
        w = {d: beats(_vals(runs, "gate_spatial", d, "AUDC_rel"),
                      _vals(runs, "gate_global", d, "AUDC_rel")) for d in ["cloud", *others]}
        ev = "; ".join(f"{d}: " + _fmt("spatial", _vals(runs, "gate_spatial", d, "AUDC_rel"))
                       + " vs " + _fmt("global", _vals(runs, "gate_global", d, "AUDC_rel"))
                       for d in w)
        if any(x is None for x in w.values()):
            verdict, action = "inconclusive", ""
        elif w["cloud"] and not any(w[d] for d in others):
            verdict, action = "held", "headline: carry gate_spatial to Ch. 5/6"
        elif not w["cloud"]:
            verdict, action = "rejected", "granularity does not help under cloud: negative result"
        elif all(w[d] for d in others):
            verdict = "rejected"
            action = ("granularity helps uniformly (§4 rejection; §5 calls this partial). "
                      "Gate arms are parameter-matched by construction")
        else:
            verdict, action = "partial", "spatial also wins under: " + ", ".join(
                d for d in others if w[d])
        add("H4", "gate_spatial vs gate_global, AUDC_rel", verdict, ev, action)

    # H5 architecture over capacity: params do not predict AUDC.
    t = runs[runs["degradation"] != "none"].groupby(
        ["model", "degradation"], as_index=False)["AUDC"].mean()
    params = runs.attrs.get("params", {})
    if params and t["model"].nunique() >= 3:
        rhos = {}
        for deg, g in t.groupby("degradation"):
            g = g[g["model"].isin(params)]
            if len(g) >= 3:
                p = g["model"].map(params).astype(float)
                rhos[deg] = p.rank().corr(g["AUDC"].rank())
        rejected = any(abs(r) >= H5_RHO for r in rhos.values() if not np.isnan(r))
        add("H5", f"|Spearman(params, AUDC)| < {H5_RHO}",
            "rejected" if rejected else ("held" if rhos else "inconclusive"),
            "; ".join(f"{d} rho={r:.2f}" for d, r in rhos.items()),
            "capacity confound: parameter-match or claim on AUDC_rel only" if rejected else "")

    # H6 temporal misalignment is unresponsive to gating.
    if "temporal" in degs and "early" in models and set(GATES) & models:
        early = _vals(runs, "early", "temporal", "AUDC_rel")
        b = {g: beats(_vals(runs, g, "temporal", "AUDC_rel"), early)
             for g in GATES if g in models}
        verdict = _verdict(all(x is False for x in b.values()),
                           any(x is True for x in b.values()),
                           any(x is None for x in b.values()))
        add("H6", "AUDC_rel under temporal", verdict,
            "; ".join([_fmt("early", early)] + [
                _fmt(g, _vals(runs, g, "temporal", "AUDC_rel")) for g in b]),
            "assume a broken simulation first" if verdict == "rejected" else "")
    return pd.DataFrame(out, columns=["hypothesis", "scope", "verdict", "evidence", "action"])


def separable(runs: pd.DataFrame) -> bool | None:
    """False when NO pair of configs separates (beyond seed spread) on clean F1 or on any
    degradation's AUDC_rel: the §5 hard stop. None if seeds are too few to tell."""
    tests = [("clean", None)] + [("AUDC_rel", d) for d in sorted(set(runs["degradation"]))]
    models = sorted(set(runs["model"]))
    seen_none = False
    for col, deg in tests:
        for a in models:
            for b in models:
                if a == b:
                    continue
                r = beats(_vals(runs, a, deg, col), _vals(runs, b, deg, col))
                if r:
                    return True
                seen_none |= r is None
    return None if seen_none else False


# ---------------------------------------------------------------- figures

def _spread_labels(ends: list[list], min_gap: float) -> list[list]:
    """Nudge end-label y positions apart so converging lines keep readable labels."""
    ends = sorted(ends, key=lambda e: e[0])
    for i in range(1, len(ends)):
        ends[i][0] = max(ends[i][0], ends[i - 1][0] + min_gap)
    return ends


def _save(fig, path: Path, **layout) -> None:
    fig.tight_layout(**layout)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def _style(ax) -> None:
    ax.set_facecolor(SURFACE)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.tick_params(colors=INK_MUTED, labelsize=8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)


def plot_panels(cv: pd.DataFrame, col: str, ylabel: str, title: str, path: Path,
                colors: dict[str, str], ylim: tuple[float, float] | None = None) -> None:
    """One panel per degradation, one line per config: seed mean, seed min-max band."""
    degs = [d for d in DEGRADATIONS if d in set(cv["degradation"])] + sorted(
        set(cv["degradation"]) - set(DEGRADATIONS))
    fig, axes = plt.subplots(1, len(degs), figsize=(3.4 * len(degs) + 0.6, 3.6), dpi=150,
                             squeeze=False)
    fig.patch.set_facecolor(SURFACE)
    for ax, deg in zip(axes[0], degs):
        d = cv[cv["degradation"] == deg]
        s = d.groupby(["model", "severity"])[col].agg(["mean", "min", "max"]).reset_index()
        ends = []
        for model, g in s.groupby("model"):
            g = g.sort_values("severity")
            ax.fill_between(g["severity"], g["min"], g["max"], color=colors[model],
                            alpha=0.15, linewidth=0, zorder=2)
            ax.plot(g["severity"], g["mean"], color=colors[model], linewidth=2, marker="o",
                    markersize=5, markeredgecolor=SURFACE, markeredgewidth=1.5, zorder=3,
                    clip_on=False, label=model)
            ends.append([g["mean"].iloc[-1], model, g["severity"].iloc[-1]])
        if ylim:
            ax.set_ylim(*ylim)
        lo, hi = ax.get_ylim()
        for y, model, x in _spread_labels(ends, min_gap=(hi - lo) * 0.06):
            ax.annotate(model, (x, y), xytext=(6, 0), textcoords="offset points",
                        va="center", fontsize=7, color=INK_MUTED, annotation_clip=False)
        sev = sorted(d["severity"].unique())
        ax.set_xticks(sev)
        ax.set_xlim(min(sev), max(sev) + (max(sev) - min(sev) or 1) * 0.35)
        ax.set_title(deg, color=INK, loc="left", fontsize=10)
        ax.set_xlabel("severity" if deg != "temporal" else "grid steps", color=INK_MUTED,
                      fontsize=8)
        _style(ax)
    axes[0][0].set_ylabel(ylabel, color=INK_MUTED)
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=8, labelcolor=INK_MUTED,
               loc="upper right", ncol=len(labels))
    fig.suptitle(title, color=INK, x=0.01, ha="left", fontsize=11)
    _save(fig, path, rect=(0, 0, 1, 0.9))


def plot_h4(runs: pd.DataFrame, path: Path, colors: dict[str, str]) -> None:
    """AUDC_rel of gate_global vs gate_spatial per degradation: seed mean dot, min-max bar."""
    degs = [d for d in DEGRADATIONS if d in set(runs["degradation"])]
    fig, ax = plt.subplots(figsize=(1.3 * len(degs) + 2.5, 3.6), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    for k, model in enumerate(GATES):
        g = runs[runs["model"] == model]
        for i, deg in enumerate(degs):
            v = g[g["degradation"] == deg]["AUDC_rel"]
            if v.empty:
                continue
            x = i + (k - 0.5) * 0.25
            ax.vlines(x, v.min(), v.max(), color=colors[model], linewidth=2, zorder=2)
            ax.plot(x, v.mean(), "o", color=colors[model], markersize=8,
                    markeredgecolor=SURFACE, markeredgewidth=1.5, zorder=3,
                    label=model if i == 0 else None)
    ax.set_xticks(range(len(degs)), degs)
    ax.set_ylabel("AUDC_rel (mean retention)", color=INK_MUTED)
    ax.set_title("H4: gate granularity x degradation structure", color=INK, loc="left",
                 fontsize=11)
    _style(ax)
    ax.legend(frameon=False, fontsize=8, labelcolor=INK_MUTED, loc="lower right",
              bbox_to_anchor=(1.0, 1.0), ncol=2, borderaxespad=0.2)
    _save(fig, path)


# ---------------------------------------------------------------- entrypoint

def analyse(root: Path, out: Path, metric: str = METRIC) -> dict[str, Path]:
    df = load_results(root)
    cv = curves(df, metric)
    runs = per_run(df, cv, metric)
    runs.attrs["params"] = (df.drop_duplicates("model").set_index("model")["params"]
                            .dropna().to_dict())
    out.mkdir(parents=True, exist_ok=True)
    t1, t2 = table1(df, runs), hypotheses(runs)
    files = {"table1": out / "table1.csv", "table2": out / "table2_hypotheses.csv",
             "orderings": out / "orderings.csv", "per_run": out / "per_run.csv",
             "curves": out / "curves.csv"}
    t1.to_csv(files["table1"], index=False)
    t2.to_csv(files["table2"], index=False)
    orderings(t1).to_csv(files["orderings"], index=False)
    runs.to_csv(files["per_run"], index=False)
    cv.to_csv(files["curves"], index=False)
    colors = model_colors(df["model"].tolist())
    if not cv.empty:
        files["fig1"] = out / "fig1_absolute.png"
        files["fig2"] = out / "fig2_retention.png"
        plot_panels(cv, "F1", metric, f"Figure 1 - absolute {metric} under degradation",
                    files["fig1"], colors)
        plot_panels(cv, "R", "R(s) = F1(s) / F1(0)",
                    "Figure 2 - relative retention (headline)", files["fig2"], colors)
        if set(GATES) <= set(runs["model"]):
            files["fig3"] = out / "fig3_h4_interaction.png"
            plot_h4(runs, files["fig3"], colors)
    sep = separable(runs)
    (out / "summary.txt").write_text(
        f"runs: {df['run_id'].nunique()}  configs: {sorted(df['model'].unique())}\n"
        f"degradations: {sorted(cv['degradation'].unique()) if not cv.empty else []}\n"
        f"configs separable beyond seed spread: {sep}"
        + ("  ** HARD STOP (§5): degradation protocol too weak **" if sep is False else "")
        + "\n\n" + t2.to_string(index=False) + "\n")
    files["summary"] = out / "summary.txt"
    return files


def aggregate(root: Path, out: Path, metric: str = METRIC) -> tuple[pd.DataFrame, list[Path]]:
    df = load_results(root)
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "results_all.csv", index=False)
    colors = model_colors(df["model"].tolist())
    plots = []
    for deg in sorted(set(df["degradation"]) - {"none"}):
        plots.append(out / f"curve_{deg}_{metric}.png")
        plot_panels(df[df["degradation"] == deg], metric, metric, f"{metric} by severity",
                    plots[-1], colors, ylim=(0, 1))
    return df, plots


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["aggregate", "exp1"])
    ap.add_argument("root", type=Path, help="directory searched for <run>/results/*.csv")
    ap.add_argument("--out", type=Path, default=None,
                    help="default: <root>/aggregate or <root>/analysis")
    ap.add_argument("--metric", default=METRIC, choices=METRICS)
    args = ap.parse_args()
    if args.command == "aggregate":
        df, plots = aggregate(args.root, args.out or args.root / "aggregate", args.metric)
        print(f"{len(df)} rows; plots: {', '.join(str(p) for p in plots) or 'none'}")
        return
    files = analyse(args.root, args.out or args.root / "analysis", args.metric)
    print(files["summary"].read_text())
    print("\n".join(f"{k}: {v}" for k, v in files.items()))


if __name__ == "__main__":
    main()
