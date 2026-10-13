"""Experiment 1 analysis on constructed results whose verdicts are known in advance."""
from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest

from src.analysis import analyse, beats, curves, hypotheses, per_run, separable
from src.results import ResultsRow, append_rows

GRIDS = {"cloud": [0.25, 0.5, 0.75, 1.0], "speckle": [0.25, 0.5, 0.75, 1.0],
         "drop_optical": [1.0], "drop_sar": [1.0], "temporal": [1, 2, 4, 8]}
CLEAN = {"opt": 0.70, "sar": 0.45, "early": 0.70, "late": 0.70,
         "gate_global": 0.70, "gate_spatial": 0.70}
PARAMS = {"opt": 21, "sar": 21, "early": 40, "late": 11, "gate_global": 30, "gate_spatial": 12}
# Retention at severity 1 per (config, degradation): the "true" story H1-H6 all hold.
SLOPE = {
    "cloud": {"opt": .2, "sar": 1, "early": .3, "late": .5, "gate_global": .6,
              "gate_spatial": .8},
    "speckle": {"opt": 1, "sar": .5, "early": .9, "late": .92, "gate_global": .94,
                "gate_spatial": .94},
    "drop_optical": {"opt": .2, "sar": 1, "early": .3, "late": .6, "gate_global": .7,
                     "gate_spatial": .7},
    "drop_sar": {"opt": 1, "sar": .2, "early": .9, "late": .95, "gate_global": .96,
                 "gate_spatial": .96},
    "temporal": {"opt": 1, "sar": .6, "early": .85, "late": .86, "gate_global": .85,
                 "gate_spatial": .85},
}


def _rows(seeds=(0, 1, 2), slope=SLOPE) -> list[ResultsRow]:
    base = ResultsRow(run_id="", git_sha="x", config_hash="h", timestamp="", model="",
                      data="d", degradation="none", severity=0.0, seed=0, split="test",
                      ckpt_kind="best", threshold=0.5, f1_macro=0.0, f1_micro=0.0,
                      accuracy=0.0, ece=None, params=0, latency_ms=1.0, flops=100.0)
    rows = []
    for model, clean in CLEAN.items():
        for seed in seeds:
            jitter = 0.002 * (seed - 1)
            rid = f"{model}-{seed}"
            r = replace(base, run_id=rid, model=model, seed=seed, params=PARAMS[model],
                        timestamp=f"2026-01-01T00:00:{seed:02d}")
            rows.append(replace(r, f1_macro=clean + jitter))
            for deg, grid in GRIDS.items():
                top = max(grid)
                for s in grid:
                    ret = 1 - (1 - slope[deg][model]) * s / top
                    rows.append(replace(r, degradation=deg, severity=float(s),
                                        f1_macro=(clean + jitter) * ret))
    return rows


def _runs(rows, params=PARAMS) -> pd.DataFrame:
    from src.results import FIELDNAMES

    df = pd.DataFrame([r.__dict__ for r in rows])[FIELDNAMES]
    runs = per_run(df, curves(df))
    runs.attrs["params"] = params
    return runs


def _verdicts(runs) -> dict[str, list[str]]:
    t = hypotheses(runs)
    return t.groupby("hypothesis")["verdict"].apply(list).to_dict()


def test_beats_needs_disjoint_ranges_and_two_seeds() -> None:
    import numpy as np

    assert beats(np.array([3.0, 4.0]), np.array([1.0, 2.0])) is True
    assert beats(np.array([2.0, 4.0]), np.array([1.0, 2.5])) is False
    assert beats(np.array([3.0]), np.array([1.0, 2.0])) is None


def test_curves_insert_clean_point_and_retention() -> None:
    from src.results import FIELDNAMES

    df = pd.DataFrame([r.__dict__ for r in _rows()])[FIELDNAMES]
    cv = curves(df)
    c = cv[(cv["run_id"] == "gate_spatial-1") & (cv["degradation"] == "cloud")]
    assert c["severity"].tolist() == [0.0, 0.25, 0.5, 0.75, 1.0]
    assert c["R"].iloc[0] == 1.0 and c["R"].iloc[-1] == pytest.approx(0.8)


def test_story_where_every_hypothesis_holds() -> None:
    v = _verdicts(_runs(_rows()))
    assert v == {"H1": ["held"], "H2": ["held"], "H3": ["held", "held"], "H4": ["held"],
                 "H5": ["held"], "H6": ["held"]}


def test_h4_uniform_win_and_h6_gate_advantage_are_rejections() -> None:
    slope = {d: dict(m) for d, m in SLOPE.items()}
    slope["speckle"]["gate_spatial"] = 0.99
    slope["drop_optical"]["gate_spatial"] = 0.8
    slope["drop_sar"]["gate_spatial"] = 0.99
    slope["temporal"]["gate_spatial"] = 0.99
    v = _verdicts(_runs(_rows(slope=slope)))
    assert v["H4"] == ["rejected"] and v["H6"] == ["rejected"]


def test_h5_rejected_when_params_track_audc() -> None:
    params = {"opt": 30, "sar": 11, "early": 12, "late": 40, "gate_global": 21,
              "gate_spatial": 21}  # temporal: Spearman rho 0.89
    assert _verdicts(_runs(_rows(), params))["H5"] == ["rejected"]


def test_single_seed_is_inconclusive_not_held() -> None:
    v = _verdicts(_runs(_rows(seeds=(0,))))
    assert v["H4"] == ["inconclusive"] and v["H1"] == ["inconclusive"]


def test_hard_stop_when_nothing_separates() -> None:
    flat = {d: {m: 1.0 for m in CLEAN} for d in SLOPE}
    rows = [replace(r, f1_macro=0.5 + 0.01 * r.seed) for r in _rows(slope=flat)]
    assert separable(_runs(rows)) is False
    assert separable(_runs(_rows())) is True


def test_analyse_writes_tables_and_figures(tmp_path: Path) -> None:
    rows = _rows()
    for rid in {r.run_id for r in rows}:
        append_rows([r for r in rows if r.run_id == rid], tmp_path / rid / "results" / "e.csv")
    files = analyse(tmp_path, tmp_path / "analysis")
    for k in ("table1", "table2", "orderings", "fig1", "fig2", "fig3", "summary"):
        assert files[k].exists() and files[k].stat().st_size > 0, k
    t1 = pd.read_csv(files["table1"])
    assert set(t1["degradation"]) == {"none", *GRIDS}
    assert (t1["n_seeds"] == 3).all()
    assert "HARD STOP" not in files["summary"].read_text()
