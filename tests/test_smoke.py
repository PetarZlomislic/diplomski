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
    (results,) = tmp_path.glob("*/results/*.csv")
    rows = results.read_text().splitlines()
    assert len(rows) == 2, rows  # header + one clean test-split row
    assert ",none,0.0,0,test,best," in rows[1]


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
    newest = max((run_dir / "results").glob("*.csv"), key=lambda p: p.stat().st_mtime)
    header = newest.read_text().splitlines()[0]
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
    sweep = df[df["degradation"] == "gaussian_noise"]
    assert len(sweep) == 10
    assert sweep.groupby("model").size().to_dict() == {"dummy": 5, "dummy_wide": 5}
    assert [p.name for p in plots] == ["curve_gaussian_noise_f1_macro.png"]
    assert plots[0].stat().st_size > 0


def test_resume_from_checkpoint_continues_same_run(tmp_path) -> None:
    import torch

    from src.train import run
    from tests.conftest import compose_cfg

    first = run(compose_cfg("+experiment=smoke", f"output_dir={tmp_path}"), {"model": "dummy"})
    last = first / "checkpoints" / "last.ckpt"
    steps_one_epoch = torch.load(last, weights_only=False)["global_step"]
    assert steps_one_epoch > 0
    cfg = compose_cfg(
        "+experiment=smoke", "trainer.max_epochs=2", f"resume_from_checkpoint={last.as_posix()}"
    )
    second = run(cfg, {"model": "dummy"})
    assert second == first
    assert len(list(tmp_path.iterdir())) == 1
    assert torch.load(last, weights_only=False)["global_step"] == 2 * steps_one_epoch


def test_time_budget_arithmetic() -> None:
    from src.train import time_budget_s

    assert time_budget_s(None, 100, 900) is None
    assert time_budget_s(9 * 3600, 600, 900) == 9 * 3600 - 600 - 900
    assert time_budget_s(3600, 3500, 900) == 60.0  # late runs still get time to checkpoint


def test_time_budget_stops_training_and_saves_resumable_checkpoint(tmp_path, monkeypatch) -> None:
    import json

    import torch

    import src.train as train_mod
    from tests.conftest import compose_cfg

    monkeypatch.setattr(train_mod, "time_budget_s", lambda *a: 2.0)
    cfg = compose_cfg("+experiment=smoke", "trainer.max_epochs=100000", f"output_dir={tmp_path}")
    run_dir = train_mod.run(cfg, {"model": "dummy"})
    events = [json.loads(x) for x in (run_dir / "run.jsonl").read_text().splitlines()]
    (stop,) = [e for e in events if e["event"] == "time_budget_stop"]
    assert stop["epoch"] < 100000 and "resume_from_checkpoint=" in stop["resume"]
    last = torch.load(run_dir / "checkpoints" / "last.ckpt", weights_only=False)
    assert last["global_step"] == stop["global_step"]
    assert "run_end" in [e["event"] for e in events]


def _failing_upload(monkeypatch) -> list:
    import huggingface_hub

    calls: list = []

    def boom(self, **kwargs):
        calls.append(kwargs)
        raise ConnectionError("simulated hub outage")

    monkeypatch.setattr(huggingface_hub.HfApi, "upload_file", boom)
    return calls


def test_push_rows_failure_warns_instead_of_raising(tmp_path, monkeypatch, caplog) -> None:
    from src.results import push_rows

    calls = _failing_upload(monkeypatch)
    csv = tmp_path / "run" / "results" / "20260101-000000-abcdef.csv"
    csv.parent.mkdir(parents=True)
    csv.write_text("run_id\nx\n")
    with caplog.at_level("WARNING"):
        assert push_rows(csv, "acme/results", token=None) is False
    assert len(calls) == 1 and calls[0]["path_in_repo"] == "results/run/20260101-000000-abcdef.csv"
    assert "simulated hub outage" in caplog.text


def test_evaluate_survives_failed_push_and_logs_it(trained_run, monkeypatch) -> None:
    import json

    calls = _failing_upload(monkeypatch)
    monkeypatch.setenv("RUN_ENV", "local")
    monkeypatch.setenv("HAS_INTERNET", "1")
    run_dir = trained_run("dummy")
    from src.evaluate import evaluate
    from tests.conftest import compose_cfg

    cfg = compose_cfg("degradation=gaussian_noise")
    rows = evaluate(str(run_dir), "best", cfg.degradation, "gaussian_noise", [0.0, 1.0],
                    cfg.degradation_seed, results_repo="acme/results")
    assert len(rows) == 2 and len(calls) == 1
    events = [json.loads(x) for x in (run_dir / "run.jsonl").read_text().splitlines()]
    end = [e for e in events if e["event"] == "run_end"][-1]
    assert end["results_pushed"] is False
    assert any(e["event"] == "log" and "simulated hub outage" in e["message"] for e in events)


def test_no_push_attempted_without_internet(trained_run, monkeypatch) -> None:
    calls = _failing_upload(monkeypatch)
    monkeypatch.setenv("RUN_ENV", "local")
    monkeypatch.setenv("HAS_INTERNET", "0")
    from src.evaluate import evaluate
    from tests.conftest import compose_cfg

    cfg = compose_cfg()
    evaluate(str(trained_run("dummy")), "best", cfg.degradation, "none", [0.0],
             cfg.degradation_seed, results_repo="acme/results")
    assert calls == []


def test_each_evaluation_writes_its_own_results_file(trained_run) -> None:
    run_dir = trained_run("dummy")
    before = set((run_dir / "results").glob("*.csv"))
    _eval(run_dir, "gaussian_noise", [0.0, 1.0])
    _eval(run_dir, "gaussian_noise", [0.0, 1.0])
    new = set((run_dir / "results").glob("*.csv")) - before
    assert len(new) == 2  # parallel evaluations of one run never share a file
    assert all(len(p.read_text().splitlines()) == 3 for p in new)  # header + 2 rows


def test_clean_eval_falls_back_to_last_when_no_best(tmp_path) -> None:
    import csv

    import src.train as train_mod
    from tests.conftest import compose_cfg

    # No validation -> val/f1_macro is never logged -> ModelCheckpoint never writes best.ckpt.
    cfg = compose_cfg("+experiment=smoke", "+trainer.limit_val_batches=0",
                      f"output_dir={tmp_path}")
    run_dir = train_mod.run(cfg, {"model": "dummy"})
    assert not (run_dir / "checkpoints" / "best.ckpt").exists()
    (results,) = (run_dir / "results").glob("*.csv")
    (row,) = list(csv.DictReader(results.open()))
    assert row["ckpt_kind"] == "last"
