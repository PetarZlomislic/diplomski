import importlib.util
import json
import os
import shlex
import subprocess
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
OVERRIDES = ["model=dummy_wide", "optim.lr=3e-4", "data.in_channels={optical: 12, sar: 2}"]
GIT = ["git", "-c", "user.name=t", "-c", "user.email=t@example.com", "-c", "core.autocrlf=false"]


def _load_submit() -> types.ModuleType:
    spec = importlib.util.spec_from_file_location("submit", ROOT / "scripts" / "submit.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    r = tmp_path / "repo"
    (r / "scripts").mkdir(parents=True)
    (r / "scripts" / "bootstrap.py").write_text((ROOT / "scripts" / "bootstrap.py").read_text())
    (r / ".gitignore").write_text("/outputs/\n.env\n")
    subprocess.run([*GIT, "init", "-q"], cwd=r, check=True)
    subprocess.run([*GIT, "add", "-A"], cwd=r, check=True)
    subprocess.run([*GIT, "commit", "-qm", "init"], cwd=r, check=True)
    subprocess.run(["git", "remote", "add", "origin", "git@github.com:acme/robustness-eval.git"],
                   cwd=r, check=True)
    env = {k: v for k, v in os.environ.items() if k not in ("REPO_SLUG", "WANDB_API_KEY")}
    monkeypatch.setattr(os, "environ", env | {"HF_BUCKET": "acme/ckpts",
                                              "KAGGLE_KERNEL": "acme/rob-train"})
    return r


@pytest.fixture
def submit(repo: Path, monkeypatch: pytest.MonkeyPatch, capsys):
    mod = _load_submit()
    monkeypatch.setattr(mod, "ROOT", repo)

    def run(*args: str, **env: str | None) -> str:
        for k, v in env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        assert mod.main(list(args)) == 0
        return capsys.readouterr().out

    run.mod = mod
    return run


def _hf_commands(stdout: str) -> list[list[str]]:
    return [shlex.split(line) for line in stdout.splitlines() if line.startswith("hf jobs")]


def _sha(repo: Path) -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True,
                          text=True, check=True).stdout.strip()


def _has(cmd: list[str], *pair: str) -> bool:
    return any(tuple(cmd[i:i + len(pair)]) == pair for i in range(len(cmd)))


def test_hf_dry_run_passes_every_override_verbatim(submit, repo: Path) -> None:
    (cmd,) = _hf_commands(submit("hf", "--dry-run", *OVERRIDES, WANDB_API_KEY="dummy"))
    assert cmd[cmd.index("--") + 1:] == OVERRIDES
    raw = "https://raw.githubusercontent.com/acme/robustness-eval"
    assert f"{raw}/{_sha(repo)}/scripts/bootstrap.py" in cmd
    for pair in (("-s", "HF_TOKEN"), ("-s", "WANDB_API_KEY"), ("-v", "hf://buckets/acme/ckpts/ckpt:/ckpt"),
                 ("--flavor", "a10g-small"), ("--timeout", "4h"), ("-e", "RUN_ENV=hf_jobs"),
                 ("-e", "MAX_RUNTIME_S=14400")):
        assert _has(cmd, *pair), pair
    assert "-d" in cmd


def test_flags_after_overrides_and_verbatim_tail(submit) -> None:
    (cmd,) = _hf_commands(submit("hf", "model=x", "--dry-run", "--entry", "prepare_data",
                                 "--", "--source", "reben"))
    assert cmd[cmd.index("--") + 1:] == ["model=x", "--source", "reben"]
    assert _has(cmd, "--flavor", "cpu-xl") and _has(cmd, "--timeout", "24h")
    assert _has(cmd, "-e", "ENTRY=prepare_data")


def test_dirty_tree_blocks_and_allow_dirty_bypasses(submit, repo: Path) -> None:
    (repo / "scripts" / "bootstrap.py").write_text("# changed\n")
    for target in ("hf", "kaggle"):
        with pytest.raises(SystemExit, match="dirty"):
            submit.mod.main([target, "--dry-run", "model=dummy"])
        submit(target, "--dry-run", "--allow-dirty", "model=dummy")


@pytest.mark.parametrize("url", [
    "https://github.com/acme/robustness-eval.git",
    "https://someone@github.com/acme/robustness-eval.git",
    "https://github.com/acme/robustness-eval",
    "ssh://git@github.com/acme/robustness-eval.git",
])
def test_repo_slug_parses_every_github_remote_form(submit, repo: Path, url: str) -> None:
    subprocess.run(["git", "remote", "set-url", "origin", url], cwd=repo, check=True)
    (cmd,) = _hf_commands(submit("hf", "--dry-run", "model=dummy"))
    assert "REPO_SLUG=acme/robustness-eval" in cmd


def test_sweep_over_2x2_emits_exactly_4_distinct_commands(submit) -> None:
    cmds = _hf_commands(submit("hf", "--sweep", "--dry-run", "model=dummy,dummy_wide",
                               "seed=0,1", "severities=[0.0,0.5]"))
    tails = {tuple(c[c.index("--") + 1:]) for c in cmds}
    assert len(cmds) == 4 and len(tails) == 4
    assert {(t[0], t[1]) for t in tails} == {
        ("model=dummy", "seed=0"), ("model=dummy", "seed=1"),
        ("model=dummy_wide", "seed=0"), ("model=dummy_wide", "seed=1"),
    }
    assert all(t[2] == "severities=[0.0,0.5]" for t in tails)


def test_kaggle_dry_run_stamps_kernel_and_overrides_round_trip(submit, repo: Path) -> None:
    out = submit("kaggle", "--dry-run", *OVERRIDES)
    assert "kaggle kernels push -p" in out and "RUN_ENV=kaggle" in out
    build = repo / "outputs" / "kaggle_build" / "acme_rob-train"
    ns: dict = {}
    exec(compile((build / "run.py").read_text(), "run.py", "exec"), ns)
    stamped = ns["STAMPED"]
    assert stamped["REPO_SLUG"] == "acme/robustness-eval" and stamped["REPO_SHA"] == _sha(repo)
    assert shlex.split(stamped["HYDRA_OVERRIDES"]) == OVERRIDES
    meta = json.loads((build / "kernel-metadata.json").read_text())
    assert meta["id"] == "acme/rob-train" and meta["kernel_type"] == "script"
    assert meta["enable_gpu"] is True and meta["enable_internet"] is True


def test_dotenv_is_loaded_and_shell_env_wins(submit, repo: Path) -> None:
    (repo / ".env").write_text(
        "# comment\nHF_BUCKET=\"dotenv/bucket\"\nexport KAGGLE_KERNEL='dotenv/kernel'\n"
        "HF_TOKEN=acme/real   # inline comment\nWANDB_ENTITY='acme/k # kept'\n")
    out = submit("hf", "--dry-run", "--allow-dirty", "m=1", HF_BUCKET=None, KAGGLE_KERNEL=None,
                 HF_TOKEN=None, WANDB_ENTITY=None)
    assert "hf://buckets/dotenv/bucket/ckpt:/ckpt" in _hf_commands(out)[0]
    assert os.environ["HF_TOKEN"] == "acme/real" and os.environ["WANDB_ENTITY"] == "acme/k # kept"
    assert "# kernel dotenv/kernel " in submit("kaggle", "--dry-run", "--allow-dirty", "m=1")
    out = submit("hf", "--dry-run", "--allow-dirty", "m=1", HF_BUCKET="shell/bucket")
    assert "hf://buckets/shell/bucket/ckpt:/ckpt" in _hf_commands(out)[0]


def test_wandb_secret_forwarded_only_when_set(submit) -> None:
    (cmd,) = _hf_commands(submit("hf", "--dry-run", "m=1", WANDB_API_KEY=None))
    assert "WANDB_API_KEY" not in cmd and cmd[cmd.index("-s") + 1] == "HF_TOKEN"


def test_real_submission_parses_namespaced_job_id_from_json(submit, monkeypatch) -> None:
    calls = []
    out_json = ('{"message": "Job started", "id": "6ac758e4df2184ac91acab1e", "url": '
                '"https://huggingface.co/jobs/peroz1/6ac758e4df2184ac91acab1e"}')

    real_run = subprocess.run

    def fake_run(cmd, **kw):
        if cmd[0] == "git":
            return real_run(cmd, **kw)
        calls.append(cmd)  # the pinned commit is a 40-hex string in cmd: must not be taken
        return subprocess.CompletedProcess(cmd, 0, f"args: {' '.join(cmd)}\n{out_json}\n", "")

    monkeypatch.setattr(submit.mod.subprocess, "run", fake_run)
    out = submit("hf", "model=dummy")
    assert "--json" in calls[0]
    assert "job_id=peroz1/6ac758e4df2184ac91acab1e" in out
    assert "hf jobs logs -f peroz1/6ac758e4df2184ac91acab1e" in out


def test_relative_data_uri_is_downloaded_into_the_job(submit) -> None:
    (cmd,) = _hf_commands(submit("hf", "--dry-run", "data=streaming",
                                 "data.data_uri=acme/reben-sweep"))
    assert "DATASETS=acme/reben-sweep" in cmd
    (cmd,) = _hf_commands(submit("hf", "--dry-run", "--entry", "evaluate", "--data",
                                 "acme/reben-sweep", "--data", "acme/other", "run_dir=x"))
    assert "DATASETS=acme/other acme/reben-sweep" in cmd
    out = submit("kaggle", "--dry-run", "--entry", "exp1_run",
                 "train=[+experiment=exp1_pilot,data.data_uri=acme/reben-sweep]")
    assert "DATASETS=acme/reben-sweep" in out
    (cmd,) = _hf_commands(submit("hf", "--dry-run", "model=dummy"))
    assert not any(c.startswith("DATASETS=") for c in cmd)


# ---------------------------------------------------------------- watch

def test_events_fold_into_job_state() -> None:
    w = _load_submit()
    st = w.JobState("o/j", "opt s0")
    lines = [
        '{"ts": "1", "event": "dataset_download_done", "gb": 41.5, "seconds": 97}',
        'noise {"ts": "2", "event": "heartbeat", "phase": "train", "epoch": 3, "step": 10, '
        '"max_steps": 20, "samples_per_s": 1300.0, "data_wait_frac": 0.4}',
        '{"ts": "3", "event": "run_end", "best_val_f1_macro": 0.57}',
        '{"ts": "4", "event": "eval_progress", "degradation": "none", "f1_macro": 0.55}',
        "Downloading bytes: 50%",
    ]
    for line in lines:
        if (e := w.parse_line(line)) is not None:
            w.apply_event(st, e)
    assert (st.epoch, st.step, st.max_steps, st.data_wait) == (3, 10, 20, 0.4)
    assert st.val_f1 == 0.57 and st.test_f1 == 0.55 and "41.5 GB" in st.note
    out = w.render([st], wait_limit=0.3, started=0.0)
    assert "0.40!" in out and "opt s0" in out


def test_jobs_from_sweep_tsv(tmp_path: Path) -> None:
    tsv = tmp_path / "s.tsv"
    tsv.write_text("job_id\toverrides\no/1\t+experiment=x model=late seed=2\n")
    assert _load_submit().jobs_from_args([str(tsv)]) == [("o/1", "late s2")]


# ---------------------------------------------------------------- auto

def _hw(name, usd_h, disk, ram="30 GB", gpu=None):
    acc = types.SimpleNamespace(model=gpu, quantity="1") if gpu else None
    return types.SimpleNamespace(name=name, unit_cost_usd=usd_h / 60, ephemeral_storage=disk,
                                 ram=ram, accelerator=acc)


class _FakeHub:
    def __init__(self, *a, **k):
        pass

    def whoami(self):
        return {"name": "acme", "isPro": True}

    def list_jobs_hardware(self):
        return [_hw("cpu-basic", 0.01, "50 GB"), _hw("t4-small", 0.4, "50 GB", gpu="T4"),
                _hw("a10g-small", 1.0, "110 GB", gpu="A10G"),
                _hw("cpu-xl", 1.0, "1000 GB", "124 GB")]

    def dataset_info(self, repo, files_metadata):
        return types.SimpleNamespace(siblings=[types.SimpleNamespace(size=60e9)])


@pytest.fixture
def fake_cloud(submit, monkeypatch):
    """Kaggle with `quota` GPU hours left and an idle kernel; a fixed HF hardware list."""
    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "HfApi", _FakeHub)
    monkeypatch.setattr(submit.mod.shutil, "which", lambda name, path=None: f"/bin/{name}")
    real_run, state = subprocess.run, {"quota": "30.00h"}

    def fake_run(cmd, **kw):
        if cmd[:2] == ["kaggle", "quota"]:
            out = json.dumps([{"resource": "GPU", "remaining": state["quota"],
                               "refreshAt": "2026-10-24T00:00:00"}])
            return subprocess.CompletedProcess(cmd, 0, out, "")
        if cmd[:3] == ["kaggle", "kernels", "status"]:
            return subprocess.CompletedProcess(cmd, 0, 'has status "COMPLETE"', "")
        return real_run(cmd, **kw)

    monkeypatch.setattr(submit.mod.subprocess, "run", fake_run)
    return state


def test_auto_picks_free_kaggle_when_quota_covers_the_job(submit, fake_cloud) -> None:
    out = submit("auto", "--dry-run", "--timeout", "8h", "model=opt")
    assert "-> kaggle" in out and "kaggle kernels push" in out and "hf jobs" not in out


def test_auto_falls_back_to_the_cheapest_hf_flavor_that_fits(submit, fake_cloud) -> None:
    fake_cloud["quota"] = "2.00h"
    out = submit("auto", "--dry-run", "--timeout", "8h", "model=opt")
    assert "GPU quota 2 h left < 8 h" in out
    (cmd,) = _hf_commands(out)
    assert _has(cmd, "--flavor", "t4-small") and not any(c.startswith("DATASETS=") for c in cmd)
    # 60 GB of data: neither Kaggle nor t4-small (50 GB disk) fits; a10g-small does.
    out = submit("auto", "--dry-run", "--timeout", "4h", "data.data_uri=acme/big")
    (cmd,) = _hf_commands(out)
    assert _has(cmd, "--flavor", "a10g-small") and _has(cmd, "-e", "MIN_FREE_GB=65")
    assert "50 GB disk < 65 GB needed" in out


def test_auto_runs_cpu_jobs_on_cpu(submit, fake_cloud, repo: Path) -> None:
    out = submit("auto", "--dry-run", "--timeout", "30m", "+experiment=smoke")
    assert "free, CPU session" in out
    meta = json.loads((repo / "outputs" / "kaggle_build" / "acme_rob-train"
                       / "kernel-metadata.json").read_text())
    assert meta["enable_gpu"] is False  # a CPU session spends no GPU quota
    (cmd,) = _hf_commands(submit("auto", "--dry-run", "--entry", "prepare_data", "--",
                                 "--source", "reben"))
    assert _has(cmd, "--flavor", "cpu-xl")  # 500 GB disk, 100 GB RAM
