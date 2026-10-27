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


# ---------------------------------------------------------------- auto / plan

PILOT = ["+experiment=exp1_pilot", "data.data_uri=acme/sweep", "data.num_workers=10"]
SIZES = {"train": 35681, "val": 18351, "test": 17974}  # the real 15 % tier


def _plan():
    import sys

    sys.path.insert(0, str(ROOT / "scripts"))
    import plan

    return plan


def test_plan_reproduces_the_measured_pilot_jobs() -> None:
    """Calibration check: HF a10g-large wall times of the 2026-10-13 pilot (running_secs)."""
    plan = _plan()
    measured = {"opt": 547, "sar": 538, "early": 554, "gate_spatial": 654}
    for arm, real in measured.items():
        run = plan.RunWork(arm, train=15 * SIZES["train"],
                           evals=15 * SIZES["val"] + SIZES["test"], workers=10)
        o = plan.Option("a10g-large", 1.5, 12, "A10G", 200, 46, plan.HF_OVERHEAD_S)
        plan.predict(plan.Work([[run]], 41.5), o, 4)
        assert abs(o.hours * 3600 / real - 1) < 0.1, (arm, o.hours * 3600, real)


def _hw(name, usd_h, vcpu, disk, ram="30 GB", gpu=None):
    acc = types.SimpleNamespace(model=gpu, quantity="1") if gpu else None
    return types.SimpleNamespace(name=name, unit_cost_usd=usd_h / 60, cpu=f"{vcpu} vCPU",
                                 ephemeral_storage=disk, ram=ram, accelerator=acc)


class _FakeHub:
    manifest: Path

    def __init__(self, *a, **k):
        pass

    def whoami(self):
        return {"name": "acme", "isPro": True}

    def list_jobs_hardware(self):
        return [_hw("cpu-basic", 0.01, 2, "50 GB", "16 GB"),
                _hw("cpu-xl", 1.0, 16, "1000 GB", "124 GB"),
                _hw("t4-small", 0.4, 4, "50 GB", "15 GB", gpu="T4"),
                _hw("a10g-large", 1.5, 12, "200 GB", "46 GB", gpu="A10G"),
                _hw("h200", 5.0, 23, "3000 GB", "256 GB", gpu="H200")]

    def dataset_info(self, repo, files_metadata):
        return types.SimpleNamespace(siblings=[types.SimpleNamespace(size=41.5e9)],
                                     private=True)

    def hf_hub_download(self, repo, filename, repo_type):
        return str(self.manifest)


@pytest.fixture
def fake_cloud(submit, monkeypatch, tmp_path):
    """Kaggle with `quota` GPU hours left and an idle kernel; a fixed HF hardware list."""
    import huggingface_hub

    _FakeHub.manifest = tmp_path / "manifest.json"
    _FakeHub.manifest.write_text(json.dumps({"sizes": SIZES}))
    monkeypatch.setattr(huggingface_hub, "HfApi", _FakeHub)
    monkeypatch.setattr(submit.mod.shutil, "which", lambda name, path=None: f"/bin/{name}")
    real_run, state = subprocess.run, {"quota": "30.00h"}

    def fake_run(cmd, **kw):
        if cmd[:2] == ["kaggle", "quota"]:
            out = "Warning: Looks like you're using an outdated `kaggle` version\n" + json.dumps(
                [{"resource": "GPU", "remaining": state["quota"],
                  "refreshAt": "2026-10-24T00:00:00"}])
            return subprocess.CompletedProcess(cmd, 0, out, "")
        if cmd[:3] == ["kaggle", "kernels", "status"]:
            state["polls"] = state.get("polls", 0) + 1
            if state["polls"] == 2:  # a network blip while polling must not end the wait
                return subprocess.CompletedProcess(cmd, 1, "", "NameResolutionError")
            return subprocess.CompletedProcess(
                cmd, 0, 'has status "KernelWorkerStatus.COMPLETE"', "")
        return real_run(cmd, **kw)

    monkeypatch.setattr(submit.mod.subprocess, "run", fake_run)
    return state


def _picked(out: str) -> str:
    return next(line.split()[1] for line in out.splitlines() if line.startswith("  -> "))


def test_auto_time_value_trades_free_kaggle_for_speed(submit, fake_cloud) -> None:
    out = submit("auto", "--dry-run", "--time-value", "0", *PILOT, "model=opt")
    assert _picked(out) == "kaggle" and "needs the Kaggle Secret HF_TOKEN" in out
    out = submit("auto", "--dry-run", "--time-value", "50", *PILOT, "model=opt")
    assert _picked(out) == "h200"
    (cmd,) = _hf_commands(out)
    assert _has(cmd, "--flavor", "h200") and _has(cmd, "-e", "MIN_FREE_GB=46")
    timeout = cmd[cmd.index("--timeout") + 1]
    assert timeout.endswith("m") and int(timeout[:-1]) < 60  # sized from the prediction


def test_auto_respects_quota_and_deadline(submit, fake_cloud) -> None:
    fake_cloud["quota"] = "0.20h"
    out = submit("auto", "--dry-run", "--time-value", "0", *PILOT, "model=opt")
    assert "GPU quota 0.2 h left" in out and _picked(out) == "t4-small"
    out = submit("auto", "--dry-run", "--time-value", "0", "--deadline", "20m", *PILOT,
                 "model=opt")
    assert "> deadline 20 min" in out and _picked(out) == "a10g-large"


def test_auto_sweep_runs_in_parallel_on_hf(submit, fake_cloud) -> None:
    out = submit("auto", "--sweep", "--dry-run", "--time-value", "0", *PILOT,
                 "model=opt,gate_spatial")
    assert "one kernel cannot run 2 jobs at once" in out
    cmds = _hf_commands(out)
    assert len(cmds) == 2 and len({c[c.index("--flavor") + 1] for c in cmds}) == 1


def test_auto_runs_cpu_jobs_on_cpu(submit, fake_cloud, repo: Path) -> None:
    # At $1/h a 30 s HF start for a fraction of a cent beats Kaggle's ~150 s free start.
    assert _picked(submit("auto", "--dry-run", "+experiment=smoke")) == "cpu-basic"
    out = submit("auto", "--dry-run", "--time-value", "0", "+experiment=smoke")
    assert ", CPU," in out and _picked(out) == "kaggle"
    meta = json.loads((repo / "outputs" / "kaggle_build" / "acme_rob-train"
                       / "kernel-metadata.json").read_text())
    assert meta["enable_gpu"] is False  # a CPU session spends no GPU quota
    (cmd,) = _hf_commands(submit("auto", "--dry-run", "--entry", "prepare_data", "--",
                                 "--source", "reben"))
    assert _has(cmd, "--flavor", "cpu-xl")  # 500 GB disk, 100 GB RAM; priced at --timeout


def test_auto_replans_without_hf_when_credit_is_used_up(submit, fake_cloud, monkeypatch) -> None:
    calls, inner = [], submit.mod.subprocess.run

    def run(cmd, **kw):
        calls.append(cmd)
        if cmd[0].endswith("hf"):
            return subprocess.CompletedProcess(cmd, 1, "", "402 Payment Required: Pre-paid "
                                               "credit balance is insufficient")
        if cmd[:3] in (["kaggle", "kernels", "push"], ["kaggle", "kernels", "logs"],
                       ["kaggle", "kernels", "output"]):
            return subprocess.CompletedProcess(cmd, 0, "", "")
        return inner(cmd, **kw)

    monkeypatch.setattr(submit.mod.subprocess, "run", run)
    monkeypatch.setattr(submit.mod.time, "sleep", lambda s: None)
    out = submit("auto", "--time-value", "50", *PILOT, "model=opt")  # would pick h200
    assert "re-planning without HF Jobs" in out and "no HF Jobs credit" in out
    assert _picked(out.split("re-planning")[1]) == "kaggle"
    assert any(c[:3] == ["kaggle", "kernels", "push"] for c in calls)
    # The blip while polling is retried; the run still ends on the real COMPLETE status.
    assert "status unavailable, retrying" in out and "acme/rob-train: COMPLETE" in out


def test_kaggle_mirror_is_attached_instead_of_downloaded(submit, repo: Path) -> None:
    out = submit("kaggle", "--dry-run", "data.data_uri=acme/sweep", "model=opt",
                 KAGGLE_MIRRORS="acme/sweep=me/sweep-copy")
    assert "KAGGLE_MIRRORS=acme/sweep=me/sweep-copy" in out
    build = repo / "outputs" / "kaggle_build" / "acme_rob-train"
    meta = json.loads((build / "kernel-metadata.json").read_text())
    assert meta["dataset_sources"] == ["me/sweep-copy"]
    ns: dict = {}
    exec(compile((build / "run.py").read_text(), "run.py", "exec"), ns)
    assert ns["STAMPED"]["DATASETS"] == "" and ns["STAMPED"]["KAGGLE_MIRRORS"] == \
        "acme/sweep=me/sweep-copy"
    # The experiment's own override is passed through unchanged.
    assert "data.data_uri=acme/sweep" in ns["STAMPED"]["HYDRA_OVERRIDES"]


def test_auto_with_a_mirror_needs_no_token_and_no_download(submit, fake_cloud) -> None:
    plain = submit("auto", "--dry-run", "--time-value", "0", *PILOT, "model=opt")
    mirrored = submit("auto", "--dry-run", "--time-value", "0", *PILOT, "model=opt",
                      KAGGLE_MIRRORS="acme/sweep=me/sweep-copy")
    assert "needs the Kaggle Secret HF_TOKEN" in plain
    assert "needs the Kaggle Secret" not in mirrored and _picked(mirrored) == "kaggle"

    def kaggle_minutes(out: str) -> float:
        cell = next(line for line in out.splitlines() if " kaggle " in line).split()[2:4]
        return float(cell[0]) * (60 if cell[1] == "h" else 1)

    assert kaggle_minutes(mirrored) < kaggle_minutes(plain)  # no 41.5 GB download
