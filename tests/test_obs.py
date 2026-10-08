import io
import json
import time
from pathlib import Path

import lightning as L
import pytest
from hydra.utils import instantiate
from omegaconf import OmegaConf

from src.callbacks.heartbeat import Heartbeat, HeartbeatCallback
from src.module import LitModule
from src.obs import RunLog
from src.train import run
from tests.conftest import compose_cfg


def _events(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


@pytest.fixture(scope="module")
def smoke_run(tmp_path_factory: pytest.TempPathFactory) -> Path:
    out = tmp_path_factory.mktemp("obs")
    return run(compose_cfg("+experiment=smoke", f"output_dir={out}"), {"model": "dummy"})


def test_every_line_of_run_jsonl_parses(smoke_run: Path) -> None:
    lines = (smoke_run / "run.jsonl").read_text(encoding="utf-8").splitlines()
    assert lines
    for line in lines:
        rec = json.loads(line)
        assert {"ts", "level", "event", "run_id"} <= rec.keys()
        assert rec["run_id"] == smoke_run.name


def test_run_start_heartbeat_run_end_present(smoke_run: Path) -> None:
    events = _events(smoke_run / "run.jsonl")
    names = [e["event"] for e in events]
    assert names[0] == "run_start" and names[-1] == "run_end"
    assert "heartbeat" in names
    start = events[0]
    for key in ("git_sha", "git_dirty", "config_hash", "resolved_config", "python", "torch",
                "lightning", "device", "cpu_count", "host_ram_gb"):
        assert key in start, key


def test_data_wait_frac_present_and_plausible(smoke_run: Path) -> None:
    (tp,) = [e for e in _events(smoke_run / "run.jsonl") if e["event"] == "throughput"]
    assert 0.0 <= tp["data_wait_frac"] < 1.0
    assert tp["samples_per_s"] > 0


def test_induced_exception_emits_run_failed_with_traceback(tmp_path: Path) -> None:
    cfg = compose_cfg("+experiment=smoke", f"output_dir={tmp_path}", "optim.lr=-1")
    with pytest.raises(Exception, match="Invalid learning rate"):
        run(cfg, {"model": "dummy"})
    (run_dir,) = tmp_path.iterdir()
    last = _events(run_dir / "run.jsonl")[-1]
    assert last["event"] == "run_failed" and last["level"] == "error"
    assert "Invalid learning rate" in last["exception"]
    assert "Traceback" in last["traceback"]
    assert last["resolved_config"]["optim"]["lr"] == -1


class _SlowFirstBatch(L.Callback):
    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx) -> None:
        if batch_idx == 0:
            time.sleep(1.5)


def test_heartbeat_fires_on_time_interval_during_a_slow_batch() -> None:
    cfg = compose_cfg("+experiment=smoke")
    sink = io.StringIO()
    hb = Heartbeat(RunLog("test", None, stream=sink), interval_s=0.25)
    module = LitModule(OmegaConf.to_container(cfg.model, resolve=True),
                       OmegaConf.to_container(cfg.optim, resolve=True), cfg.threshold)
    trainer = L.Trainer(max_epochs=1, logger=False, enable_progress_bar=False,
                        enable_model_summary=False, enable_checkpointing=False,
                        callbacks=[HeartbeatCallback(hb), _SlowFirstBatch()])
    trainer.fit(module, datamodule=instantiate(cfg.data))
    beats = [json.loads(x) for x in sink.getvalue().splitlines()]
    frozen = [b for b in beats if b["event"] == "heartbeat" and b["step"] == 0]
    assert len(frozen) >= 3, f"only {len(frozen)} heartbeats while step 0 was stuck"


def test_evaluate_emits_progress_per_severity(smoke_run: Path) -> None:
    from src.evaluate import evaluate

    cfg = compose_cfg("degradation=gaussian_noise")
    evaluate(str(smoke_run), "best", cfg.degradation, "gaussian_noise", [0.0, 0.5, 1.0],
             cfg.degradation_seed)
    progress = [e for e in _events(smoke_run / "run.jsonl")
                if e["event"] == "eval_progress" and e["degradation"] == "gaussian_noise"]
    assert [p["severity"] for p in progress] == [0.0, 0.5, 1.0]
    assert all("f1_macro" in p for p in progress)


def test_wandb_offline_writes_local_dir(tmp_path: Path) -> None:
    pytest.importorskip("wandb")
    cfg = compose_cfg("+experiment=smoke", f"output_dir={tmp_path}", "wandb.enabled=true",
                      "wandb.mode=offline")
    run_dir = run(cfg, {"model": "dummy"})
    assert list((run_dir / "wandb").glob("offline-run-*")), list(run_dir.rglob("*"))[:20]
    names = [e["event"] for e in _events(run_dir / "run.jsonl")]
    assert names[0] == "run_start" and names[-1] == "run_end"


def test_git_provenance_falls_back_to_pinned_sha(monkeypatch: pytest.MonkeyPatch) -> None:
    import src.obs as obs

    monkeypatch.setattr(obs, "_git", lambda *a: None)  # tarball checkout: no .git
    monkeypatch.delenv("REPO_SHA", raising=False)
    assert obs.git_sha() == "unknown" and obs.git_dirty() is None
    monkeypatch.setenv("REPO_SHA", "a" * 40)
    assert obs.git_sha() == "a" * 40 and obs.git_dirty() is False
