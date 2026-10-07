import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def test_train_smoke_writes_checkpoint(tmp_path: Path) -> None:
    start = time.monotonic()
    result = subprocess.run(
        [sys.executable, "-m", "src.train", "+experiment=smoke", f"output_dir={tmp_path}"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=120,
    )
    elapsed = time.monotonic() - start
    assert result.returncode == 0, result.stderr
    assert elapsed < 60, f"smoke run took {elapsed:.1f}s"
    ckpts = list(tmp_path.glob("*/checkpoints/*.ckpt"))
    names = {c.name for c in ckpts}
    assert {"best.ckpt", "last.ckpt"} <= names, names


SEVERITIES = [0.0, 0.25, 0.5, 0.75, 1.0]


def _eval(run_dir: Path, deg: str, severities: list[float]):
    from src.evaluate import evaluate
    from tests.conftest import compose_cfg

    cfg = compose_cfg(f"degradation={deg}")
    return evaluate(str(run_dir), "best", cfg.degradation, deg, severities, cfg.degradation_seed)


def test_evaluate_produces_valid_results_row(trained_run) -> None:
    from src.results import FIELDNAMES

    run_dir = trained_run("dummy")
    (row,) = _eval(run_dir, "none", [0.0])
    assert row.model == "dummy" and row.split == "test" and row.ckpt_kind == "best"
    assert 0.0 <= row.f1_macro <= 1.0 and row.params > 0
    header = (run_dir / "results.csv").read_text().splitlines()[0]
    assert header.split(",") == FIELDNAMES


def test_severity_sweep_writes_one_row_each_and_zero_equals_clean(trained_run) -> None:
    run_dir = trained_run("dummy")
    rows = _eval(run_dir, "gaussian_noise", SEVERITIES)
    assert [r.severity for r in rows] == SEVERITIES
    (clean,) = _eval(run_dir, "none", [0.0])
    for k in ("f1_macro", "f1_micro", "accuracy"):
        assert getattr(rows[0], k) == getattr(clean, k), k


def test_two_model_sweep_aggregates_to_ten_rows_and_one_curve(tmp_path) -> None:
    from src.aggregate import aggregate
    from src.train import run
    from tests.conftest import compose_cfg

    root = tmp_path / "runs"
    for model in ("dummy", "dummy_wide"):
        cfg = compose_cfg(f"model={model}", "+experiment=smoke", f"output_dir={root}")
        run_dir = run(cfg, {"model": model, "data": "dummy"})
        _eval(run_dir, "gaussian_noise", SEVERITIES)
        _eval(run_dir, "gaussian_noise", SEVERITIES)  # re-run must not double-count
    df, plots = aggregate(root, tmp_path / "agg")
    assert len(df) == 10
    assert df.groupby("model").size().to_dict() == {"dummy": 5, "dummy_wide": 5}
    assert [p.name for p in plots] == ["curve_gaussian_noise_f1_macro.png"]
    assert plots[0].stat().st_size > 0
