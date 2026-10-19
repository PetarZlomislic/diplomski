"""Run entrypoints remotely at the current commit, and watch them.

  python scripts/submit.py hf [--dry-run] [--allow-dirty] [--flavor F] [--timeout T]
                              [--entry E] [--data owner/dataset]... [--sweep] <overrides...>
  python scripts/submit.py kaggle [--dry-run] [--allow-dirty] [--entry E] [--kernel user/slug]
                                  [--data owner/dataset]... <overrides...>
  python scripts/submit.py auto [--dry-run] [--allow-dirty] [--timeout T] [--entry E]
                                [--data owner/dataset]... [--[no-]gpu] [--min-disk GB]
                                <overrides...>
  python scripts/submit.py watch [<sweep.tsv> | <owner>/<job_id>...] [--once]

auto    pick the cheapest option that can run this job right now and submit it there:
        Kaggle (free) if the kernel is idle, the job fits its 9 h GPU / 12 h CPU session and
        disk, and the weekly GPU quota (`kaggle quota`) covers --timeout; otherwise the HF
        Jobs flavor with the lowest live price x --timeout that has the GPU, disk (datasets
        + 5 GB) and RAM it needs. Prints every option with its cost or why it is out.

hf      one detached HF Job (scripts/bootstrap.py). With --sweep, one job per combination of
        comma-separated override values (model=a,b seed=0,1 -> 4 jobs; commas inside [...]
        or {...} are not split); job ids go to outputs/sweeps/<ts>.tsv. Checkpoints and logs
        go to the bucket HF_BUCKET, mounted at /ckpt. `--entry prepare_data` defaults to a
        cpu-xl flavor and 24 h, e.g. `hf --entry prepare_data -- --source reben --out-repo u/reben`.
kaggle  push a kernel running the entrypoint, wait for it, pull its outputs into
        outputs/kaggle/<kernel>/<sha>. Needs KAGGLE_KERNEL (or --kernel) and Kaggle CLI creds.
watch   live terminal dashboard for HF Jobs (newest outputs/sweeps/*.tsv by default): parses
        the JSON events the entrypoints emit, merged across polls; exits once all finished.

Hub datasets (--data, and any relative `data.data_uri=<owner>/<dataset>`) are downloaded to
the job's disk before the entrypoint (mind the flavor's disk: a10g-small has 110 GB).
Evaluation jobs read data_uri from the run config, so pass --data for them.
The tree must be clean (the job runs HEAD), unless --allow-dirty. Settings come from the
environment or a gitignored .env (the environment wins): HF_BUCKET, HF_TOKEN, WANDB_API_KEY
(forwarded when set), KAGGLE_KERNEL, REPO_SLUG (default: from the origin remote).
"""
import argparse
import concurrent.futures as cf
import itertools
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ENTRIES = ("train", "evaluate", "exp1_run", "prepare_data")
REPO = r"[A-Za-z0-9._-]+/[A-Za-z0-9._-]+"
UNITS = {"d": 86400, "h": 3600, "m": 60, "s": 1}


def die(msg: str) -> None:
    raise SystemExit(f"error: {msg}")


def load_env(path: Path) -> None:
    """Export KEY=VALUE lines of a .env without executing it. Set variables win."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        m = re.match(r"\s*(?:export\s+)?([A-Za-z_]\w*)=(.*)$", line)
        if not m or line.lstrip().startswith("#"):
            continue
        key, val = m.groups()
        q = re.fullmatch(r"\"(.*)\"|'(.*)'", val)
        val = (q[1] if q[1] is not None else q[2]) if q else re.sub(r"\s+#.*$", "", val).rstrip()
        os.environ.setdefault(key, val)


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True).stdout.strip()


def repo_slug() -> str:
    if slug := os.environ.get("REPO_SLUG"):
        return slug
    url = git("remote", "get-url", "origin")
    if not url:
        return "UNSET_OWNER/UNSET_REPO"
    url = re.sub(r"^(https?://([^@/]+@)?github\.com/|git@github\.com:|ssh://git@github\.com/)",
                 "", url).removesuffix("/").removesuffix(".git")
    if not re.fullmatch(r"[^/:@]+/[^/:@]+", url):
        die("cannot parse owner/repo from origin; set REPO_SLUG=owner/repo")
    return url


def pinned_commit(allow_dirty: bool, dry_run: bool) -> tuple[str, str]:
    """(owner/repo, HEAD sha); refuses a dirty tree, whose changes the job would not run."""
    if status := git("status", "--porcelain"):
        if not allow_dirty:
            print(status, file=sys.stderr)
            die("working tree is dirty; commit first or pass --allow-dirty "
                "(the job runs HEAD, not your local changes)")
        short = git("rev-parse", "--short", "HEAD")
        print(f"warning: working tree is dirty; submitting HEAD {short} anyway", file=sys.stderr)
    slug = repo_slug()
    if slug.startswith("UNSET_OWNER/") and not dry_run:
        die("no origin remote; set REPO_SLUG=owner/repo")
    return slug, git("rev-parse", "HEAD")


def datasets(overrides: list[str], extra: list[str]) -> str:
    repos = set(extra)
    for o in overrides:
        repos.update(re.findall(rf"data\.data_uri=({REPO})(?=[],]|$)", o))
    for r in repos:
        if not re.fullmatch(REPO, r):
            die(f"--data must be <owner>/<dataset>, got {r}")
    return " ".join(sorted(repos))


def expand(overrides: list[str]) -> list[list[str]]:
    """key=v1,v2 axes -> every combination, splitting only at bracket depth 0."""
    axes = []
    for o in overrides:
        key, sep, vals = o.partition("=")
        if not sep:
            axes.append([o])
            continue
        parts, cur, depth = [], "", 0
        for ch in vals:
            depth += (ch in "[{") - (ch in "]}")
            if ch == "," and depth == 0:
                parts.append(cur)
                cur = ""
            else:
                cur += ch
        axes.append([f"{key}={v}" for v in [*parts, cur]])
    return [list(c) for c in itertools.product(*axes)]


def default_timeout(a: argparse.Namespace) -> str:
    return a.timeout or ("24h" if a.entry == "prepare_data" else "4h")


def seconds(timeout: str) -> int:
    m = re.fullmatch(r"(\d+)([dhms]?)", timeout)
    if not m:
        die("--timeout must be an integer with optional s/m/h/d suffix")
    return int(m[1]) * UNITS.get(m[2], 1)


# ---------------------------------------------------------------- HF Jobs

def submit_hf(a: argparse.Namespace, overrides: list[str]) -> str | None:
    slug, sha = pinned_commit(a.allow_dirty, a.dry_run)
    bucket = os.environ.get("HF_BUCKET") or ("<HF_BUCKET>" if a.dry_run else None)
    if not bucket:
        die("HF_BUCKET is not set (e.g. HF_BUCKET=<user>/<bucket> in .env)")
    flavor = a.flavor or ("cpu-xl" if a.entry == "prepare_data" else "a10g-small")
    timeout = default_timeout(a)
    secrets = ["-s", "HF_TOKEN"]
    if os.environ.get("WANDB_API_KEY"):  # hf refuses to forward a secret that is unset
        secrets += ["-s", "WANDB_API_KEY"]
    data = datasets(overrides, a.data)
    cmd = ["hf", "jobs", "uv", "run", "--flavor", flavor, "--timeout", timeout, "-d", "--json",
           *secrets, "-e", "RUN_ENV=hf_jobs", "-e", f"REPO_SLUG={slug}", "-e", f"REPO_SHA={sha}",
           "-e", f"ENTRY={a.entry}", "-e", f"MAX_RUNTIME_S={seconds(timeout)}",
           *(["-e", f"DATASETS={data}"] if data else []),
           *(["-e", f"MIN_FREE_GB={a.min_free_gb}"] if getattr(a, "min_free_gb", None) else []),
           "-v", f"hf://buckets/{bucket}/ckpt:/ckpt",
           f"https://raw.githubusercontent.com/{slug}/{sha}/scripts/bootstrap.py", "--", *overrides]
    if a.dry_run:
        print(shlex.join(cmd))
        return None
    # Prefer the project's pinned hf CLI over whatever else is on PATH.
    venv = os.pathsep.join(str(ROOT / d) for d in (".venv/Scripts", ".venv/bin"))
    hf = shutil.which("hf", path=venv) or shutil.which("hf") or "hf"
    res = subprocess.run([hf, *cmd[1:]], capture_output=True, text=True)
    print(res.stdout, end="")
    if res.returncode:
        print(res.stderr, file=sys.stderr)
        die(f"hf jobs exited with {res.returncode}")
    # The --json result carries the job URL .../jobs/<owner>/<id>; take owner/id from it only,
    # never from other hex strings in the output (e.g. the pinned commit in the script URL).
    m = re.search(r'huggingface\.co/jobs/([^/"\s]+/[0-9a-f]{24})\b', res.stdout)
    if not m:
        die("could not find the job URL in the output above")
    print(f"job_id={m[1]}\nfollow: uv run hf jobs logs -f {m[1]}")
    return m[1]


def sweep_hf(a: argparse.Namespace, overrides: list[str]) -> None:
    rows = [(submit_hf(a, combo), combo) for combo in expand(overrides)]
    if not a.dry_run:
        stamp = time.strftime("%Y%m%d-%H%M%S", time.gmtime())
        manifest = ROOT / "outputs" / "sweeps" / f"{stamp}.tsv"
        manifest.parent.mkdir(parents=True, exist_ok=True)
        lines = [f"{j}\t{' '.join(c)}\n" for j, c in rows]
        manifest.write_text("job_id\toverrides\n" + "".join(lines), encoding="utf-8")
        print(f"manifest: {manifest}")


# ---------------------------------------------------------------- Kaggle

def submit_kaggle(a: argparse.Namespace, overrides: list[str]) -> int:
    slug, sha = pinned_commit(a.allow_dirty, a.dry_run)
    kernel = a.kernel or os.environ.get("KAGGLE_KERNEL")
    if not kernel:
        if not a.dry_run:
            die("set KAGGLE_KERNEL=<user>/<slug> or pass --kernel")
        kernel = f"<KAGGLE_USER>/robustness-eval-{a.entry}"
    stamp = {"REPO_SLUG": slug, "REPO_SHA": sha, "ENTRY": a.entry,
             "HYDRA_OVERRIDES": shlex.join(overrides),  # split again by src/env.py
             "DATASETS": datasets(overrides, a.data),
             "MIN_FREE_GB": str(getattr(a, "min_free_gb", None) or "")}
    safe = re.sub(r"[^A-Za-z0-9._-]", "_", kernel)
    build = ROOT / "outputs" / "kaggle_build" / safe
    build.mkdir(parents=True, exist_ok=True)
    src = (ROOT / "scripts" / "bootstrap.py").read_text(encoding="utf-8")
    (build / "run.py").write_text(src.replace("STAMPED: dict[str, str] = {}",
                                              f"STAMPED: dict[str, str] = {json.dumps(stamp)}"),
                                  encoding="utf-8")
    meta = {"id": kernel, "title": kernel.split("/")[-1], "code_file": "run.py",
            "language": "python", "kernel_type": "script", "is_private": True,
            "enable_gpu": getattr(a, "gpu", None) is not False, "enable_internet": True,
            "dataset_sources": [],
            "competition_sources": [], "kernel_sources": [], "model_sources": []}
    (build / "kernel-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    out_dir = ROOT / "outputs" / "kaggle" / safe / sha
    push = ["kaggle", "kernels", "push", "-p", str(build)]
    if a.dry_run:
        print(f"# kernel {kernel} at {slug}@{sha}, RUN_ENV=kaggle "
              f"HYDRA_OVERRIDES={stamp['HYDRA_OVERRIDES']} DATASETS={stamp['DATASETS']}")
        for c in (push, ["kaggle", "kernels", "status", kernel],
                  ["kaggle", "kernels", "output", kernel, "-p", str(out_dir)]):
            print(shlex.join(c))
        return 0
    if not shutil.which("kaggle"):
        die("kaggle CLI not found (uv tool install kaggle)")
    os.environ["PYTHONUTF8"] = "1"  # the Kaggle CLI crashes printing non-ASCII on Windows
    subprocess.run(push, check=True)
    print(f"pushed {kernel}; polling every 30s")
    while True:
        res = subprocess.run(["kaggle", "kernels", "status", kernel], capture_output=True,
                             text=True)
        status = (res.stdout + res.stderr).strip()
        print(f"{time.strftime('%H:%M:%S', time.gmtime())} {status}")
        if re.search(r"complete|error|cancel", status, re.IGNORECASE):
            break
        time.sleep(30)
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "kernel_log.json").open("w") as f:
        subprocess.run(["kaggle", "kernels", "logs", kernel], stdout=f, stderr=subprocess.STDOUT)
    try:  # `kernels output` can hang when a failed kernel produced no files; bound it
        subprocess.run(["kaggle", "kernels", "output", kernel, "-p", str(out_dir)], timeout=600)
    except subprocess.TimeoutExpired:
        print("warning: output download timed out", file=sys.stderr)
    print(f"outputs: {out_dir} (execution log: {out_dir / 'kernel_log.json'})")
    return 0 if re.search("complete", status, re.IGNORECASE) else 1


# ---------------------------------------------------------------- auto: cheapest usable option

KAGGLE_GPU_H, KAGGLE_CPU_H = 9, 12  # Kaggle session caps
KAGGLE_DISK_GB = 50  # scratch space the bootstrap has been seen to get (fits the pilot tier)
JOB_OVERHEAD_GB = 5  # repo, uv env and caches next to the downloaded datasets


@dataclass
class Option:
    name: str  # "kaggle" or an HF Jobs flavor
    usd: float | None  # estimated cost of the whole --timeout; None = unusable right now
    note: str  # spec, or why it cannot be used


@dataclass
class Needs:
    gpu: bool
    hours: float
    disk_gb: float
    ram_gb: float


def needs(a: argparse.Namespace, overrides: list[str], data_gb: float) -> Needs:
    """What the job needs. GPU unless it is data preparation or explicitly CPU (smoke)."""
    cpu_only = a.entry == "prepare_data" or any(
        o in ("trainer.accelerator=cpu", "+experiment=smoke") for o in overrides)
    prep = a.entry == "prepare_data"  # the reBEN LMDB (155 GB) + three tiers (~290 GB)
    return Needs(gpu=a.gpu if a.gpu is not None else not cpu_only,
                 hours=seconds(default_timeout(a)) / 3600,
                 disk_gb=a.min_disk or (500 if prep else data_gb + JOB_OVERHEAD_GB),
                 ram_gb=a.min_ram or (100 if prep else 0))


def _gb(spec: str | None) -> float:
    m = re.match(r"([\d.]+)\s*([GT])B", spec or "")
    return float(m[1]) * (1000 if m[2] == "T" else 1) if m else 0.0


def kaggle_option(a: argparse.Namespace, n: Needs) -> Option:
    def no(why: str) -> Option:
        return Option("kaggle", None, why)

    if a.entry == "prepare_data":
        return no("data preparation needs a big CPU machine")
    if not shutil.which("kaggle"):
        return no("kaggle CLI not found")
    kernel = a.kernel or os.environ.get("KAGGLE_KERNEL")
    if not kernel:
        return no("KAGGLE_KERNEL not set")
    cap = KAGGLE_GPU_H if n.gpu else KAGGLE_CPU_H
    if n.hours > cap:
        return no(f"timeout {n.hours:g} h > {cap} h session cap")
    if n.disk_gb > KAGGLE_DISK_GB:
        return no(f"needs {n.disk_gb:.0f} GB disk > {KAGGLE_DISK_GB} GB")
    env = os.environ | {"PYTHONUTF8": "1"}
    status = subprocess.run(["kaggle", "kernels", "status", kernel], capture_output=True,
                            text=True, env=env).stdout
    if re.search(r"running|queued", status, re.IGNORECASE):
        return no(f"{kernel} is still running (a push would replace it)")
    if n.gpu:
        res = subprocess.run(["kaggle", "quota", "--format", "json"], capture_output=True,
                             text=True, env=env)
        try:
            gpu = next(q for q in json.loads(res.stdout) if q["resource"] == "GPU")
        except (ValueError, StopIteration):
            return no("cannot read the GPU quota (`kaggle quota`)")
        left = float(gpu["remaining"].rstrip("h"))
        if left < n.hours:
            return no(f"GPU quota {left:g} h left < {n.hours:g} h (refills {gpu['refreshAt']})")
        return Option("kaggle", 0.0, f"free, T4/P100, GPU quota {left:g} h left")
    return Option("kaggle", 0.0, "free, CPU session")


def hf_options(api, n: Needs) -> list[Option]:
    """Every single-device HF Jobs flavor, priced live for the whole timeout."""
    me = api.whoami()
    blocked = None
    if not os.environ.get("HF_BUCKET"):
        blocked = "HF_BUCKET not set"
    elif not (me.get("isPro") or me.get("canPay") or me.get("orgs")):
        blocked = "account cannot run Jobs (needs PRO or billing)"
    out = []
    for h in api.list_jobs_hardware():
        acc = h.accelerator
        if (acc is not None and int(acc.quantity) > 1) or (n.gpu and acc is None):
            continue  # training uses one device; CPU flavors cannot run a GPU job
        price = h.unit_cost_usd * 60
        why = blocked
        if not why and _gb(h.ephemeral_storage) < n.disk_gb:
            why = f"{h.ephemeral_storage} disk < {n.disk_gb:.0f} GB needed"
        if not why and _gb(h.ram) < n.ram_gb:
            why = f"{h.ram} RAM < {n.ram_gb:.0f} GB needed"
        spec = (f"{price:.2f} $/h, {acc.model if acc else 'CPU'}, {h.ram} RAM, "
                f"{h.ephemeral_storage} disk")
        out.append(Option(h.name, None if why else price * n.hours, why or spec))
    return out


def auto(a: argparse.Namespace, overrides: list[str]) -> int:
    """Submit to the cheapest option usable right now (Kaggle when its quota allows: free)."""
    if a.sweep:
        die("auto submits one job; for a whole stage on Kaggle use --entry exp1_run")
    from huggingface_hub import HfApi

    api = HfApi(token=os.environ.get("HF_TOKEN"))
    repos = datasets(overrides, a.data).split()
    data_gb = sum((s.size or 0) for r in repos
                  for s in api.dataset_info(r, files_metadata=True).siblings) / 1e9
    n = needs(a, overrides, data_gb)
    options = [kaggle_option(a, n), *hf_options(api, n)]
    usable = sorted((o for o in options if o.usd is not None), key=lambda o: o.usd)
    print(f"job needs: {'GPU' if n.gpu else 'CPU'}, {n.hours:g} h, {n.disk_gb:.0f} GB disk"
          + (f", {n.ram_gb:.0f} GB RAM" if n.ram_gb else "")
          + (f" (datasets {data_gb:.1f} GB)" if repos else ""))
    for o in usable + [o for o in options if o.usd is None]:
        cost = "  n/a  " if o.usd is None else f"${o.usd:6.2f}"
        print(f"  {'->' if usable and o is usable[0] else '  '} {o.name:<16}{cost}  {o.note}")
    if not usable:
        die("nothing can run this job right now")
    best = usable[0]
    a.min_free_gb = round(n.disk_gb) if repos else None
    a.gpu = n.gpu
    if best.name == "kaggle":
        return submit_kaggle(a, overrides)
    a.flavor = best.name
    submit_hf(a, overrides)
    return 0


# ---------------------------------------------------------------- watch

TERMINAL = {"COMPLETED", "ERROR", "CANCELED", "DELETED"}
PRICE_PER_H = {"cpu-basic": 0.01, "cpu-upgrade": 0.03, "cpu-xl": 1.0, "cpu-performance": 1.9,
               "t4-small": 0.4, "t4-medium": 0.6, "l4x1": 0.8, "a10g-small": 1.0,
               "a10g-large": 1.5, "a10g-largex2": 3.0, "a100-large": 2.5, "l40sx1": 1.8}


@dataclass
class JobState:
    job_id: str
    label: str
    stage: str = "?"
    flavor: str = ""
    running_s: float = 0.0
    phase: str = ""
    epoch: int | None = None
    step: int | None = None
    max_steps: int | None = None
    samples_per_s: float | None = None
    data_wait: float | None = None
    val_f1: float | None = None
    test_f1: float | None = None
    note: str = ""
    seen: set = field(default_factory=set)


def jobs_from_args(args: list[str]) -> list[tuple[str, str]]:
    """[(job_id, label)] from explicit ids, a sweep TSV, or the newest sweep TSV."""
    if args and all("/" in a and not a.endswith(".tsv") for a in args):
        return [(a, a.split("/")[-1][:8]) for a in args]
    tsv = Path(args[0]) if args else max((ROOT / "outputs" / "sweeps").glob("*.tsv"),
                                         key=lambda p: p.stat().st_mtime, default=None)
    if tsv is None or not tsv.exists():
        sys.exit("no job ids given and no outputs/sweeps/*.tsv found")
    out = []
    for line in tsv.read_text(encoding="utf-8").splitlines()[1:]:
        if line.strip():
            job_id, overrides = (line.split("\t") + [""])[:2]
            m, s = re.search(r"model=(\S+)", overrides), re.search(r"seed=(\S+)", overrides)
            out.append((job_id, (m[1] if m else job_id[-8:]) + (f" s{s[1]}" if s else "")))
    return out


def apply_event(st: JobState, e: dict) -> None:
    """Fold one parsed log event into the job's state."""
    ev = e.get("event")
    if ev == "dataset_download_start":
        st.phase, st.note = "download", e.get("repo", "")
    elif ev == "dataset_download_done":
        st.note = f"data {e.get('gb')} GB in {e.get('seconds')} s"
    elif ev == "run_start":
        st.phase = e.get("entrypoint", "run")
    elif ev == "heartbeat":
        st.phase = e.get("phase") or st.phase
        for k, attr in (("epoch", "epoch"), ("step", "step"), ("max_steps", "max_steps"),
                        ("samples_per_s", "samples_per_s"), ("data_wait_frac", "data_wait")):
            if e.get(k) is not None:
                setattr(st, attr, e[k])
    elif ev == "run_end" and e.get("best_val_f1_macro") is not None:
        st.val_f1 = e["best_val_f1_macro"]
    elif ev == "eval_progress" and e.get("degradation") == "none":
        st.test_f1 = e.get("f1_macro")
    elif ev == "run_failed":
        st.note = "FAILED: " + str(e.get("exception", ""))[:60]
    elif ev in ("source_loaded", "tier_written", "split_pushed", "tier_pushed", "prepare_done"):
        st.phase, st.note = "prepare", ev + " " + str(e.get("split") or e.get("tier") or "")


def parse_line(line: str) -> dict | None:
    i = line.find("{")
    try:
        e = json.loads(line[i:]) if i >= 0 else None
    except json.JSONDecodeError:
        return None
    return e if isinstance(e, dict) and "event" in e else None


def poll(api, st: JobState, tail: int) -> None:
    owner, _, jid = st.job_id.rpartition("/")
    info = api.inspect_job(job_id=jid, namespace=owner or None)
    st.stage = info.status.stage if info.status else "?"
    st.flavor = info.flavor or ""
    st.running_s = float(getattr(info.durations, "running_secs", None) or 0)
    if st.stage == "SCHEDULING":
        return
    for line in api.fetch_job_logs(job_id=jid, namespace=owner or None, tail=tail):
        if "Traceback" in line and not st.note.startswith("FAILED"):
            st.note = "Traceback in log"
        e = parse_line(line)
        key = e and (e.get("ts"), e.get("event"), e.get("step"), e.get("split"))
        if e is not None and key not in st.seen:
            st.seen.add(key)
            apply_event(st, e)
    if info.status and info.status.message and st.stage == "ERROR":
        st.note = st.note or info.status.message[:60]


def fmt(v, spec: str = "") -> str:
    return "-" if v is None else format(v, spec)


def render(states: list[JobState], wait_limit: float, started: float) -> str:
    hdr = (f"{'job':<16}{'stage':<11}{'phase':<10}{'ep':>4}{'step':>12}{'smp/s':>8}"
           f"{'wait':>7}{'val F1':>8}{'test F1':>8}{'run':>8}{'$':>7}  note")
    rows, total = [hdr, "-" * len(hdr)], 0.0
    for s in states:
        cost = s.running_s / 3600 * PRICE_PER_H.get(s.flavor, 0)
        total += cost
        step = f"{s.step}/{s.max_steps}" if s.step is not None else "-"
        wait = fmt(s.data_wait, ".2f") + ("!" if (s.data_wait or 0) >= wait_limit else "")
        rows.append(f"{s.label[:15]:<16}{s.stage[:10]:<11}{s.phase[:9]:<10}"
                    f"{fmt(s.epoch):>4}{step:>12}{fmt(s.samples_per_s, '.0f'):>8}{wait:>7}"
                    f"{fmt(s.val_f1, '.3f'):>8}{fmt(s.test_f1, '.3f'):>8}"
                    f"{int(s.running_s // 60):>5} m{cost:>7.2f}  {s.note}")
    done = sum(s.stage in TERMINAL for s in states)
    rows += ["", f"{done}/{len(states)} finished   est. cost ${total:.2f}   "
             f"wait! = data_wait >= {wait_limit}   updated {time.strftime('%H:%M:%S')}   "
             f"watching {int((time.time() - started) // 60)} m   Ctrl+C to quit"]
    return "\n".join(rows)


def watch(a: argparse.Namespace) -> None:
    from huggingface_hub import HfApi

    api = HfApi(token=os.environ.get("HF_TOKEN"))
    states = [JobState(j, label) for j, label in jobs_from_args(a.jobs)]
    os.system("")  # enables ANSI escapes in the Windows console
    started = time.time()
    pool = cf.ThreadPoolExecutor(max_workers=max(1, len(states)))
    pending: dict[str, cf.Future] = {}

    def safe_poll(s: JobState) -> None:
        try:
            poll(api, s, a.tail)
        except Exception as e:  # network blips must not kill the dashboard
            s.note = f"poll error: {type(e).__name__}"

    try:
        while True:
            for s in states:
                busy = s.job_id in pending and not pending[s.job_id].done()
                if not busy and not (s.stage in TERMINAL and s.seen):
                    pending[s.job_id] = pool.submit(safe_poll, s)
            # Wait for this round, but never longer than one interval: a slow log read just
            # lands on a later refresh.
            cf.wait(list(pending.values()), timeout=a.interval)
            table = render(states, a.wait_limit, started)
            if a.once:
                print(table)
                return
            sys.stdout.write("\033[H\033[2J" + table + "\n")
            sys.stdout.flush()
            if all(s.stage in TERMINAL for s in states) and all(
                    f.done() for f in pending.values()):
                return
            time.sleep(a.interval)
    except KeyboardInterrupt:
        pass
    finally:
        pool.shutdown(wait=False, cancel_futures=True)


# ---------------------------------------------------------------- CLI

def main(argv: list[str]) -> int:
    load_env(ROOT / ".env")
    command = argv[0] if argv else ""
    ap = argparse.ArgumentParser(prog=f"submit.py {command}", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    if command == "watch":
        ap.add_argument("jobs", nargs="*", help="sweep TSV or <owner>/<job_id> ids")
        ap.add_argument("--interval", type=float, default=20, help="seconds between polls")
        ap.add_argument("--tail", type=int, default=120, help="log lines read per poll")
        ap.add_argument("--wait-limit", type=float, default=0.3)
        ap.add_argument("--once", action="store_true")
        watch(ap.parse_args(argv[1:]))
        return 0
    if command not in ("hf", "kaggle", "auto"):
        ap.error("first argument must be hf, kaggle, auto or watch")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--allow-dirty", action="store_true")
    ap.add_argument("--entry", default="train", choices=ENTRIES)
    ap.add_argument("--data", action="append", default=[], metavar="OWNER/DATASET")
    if command != "hf":
        ap.add_argument("--kernel")
        ap.add_argument("--gpu", action=argparse.BooleanOptionalAction, default=None,
                        help="Kaggle: GPU session (default on); auto: default from the job")
    if command != "kaggle":
        ap.add_argument("--timeout")
        ap.add_argument("--sweep", action="store_true")
    if command == "hf":
        ap.add_argument("--flavor")
    if command == "auto":
        ap.add_argument("--min-disk", type=float, help="GB; default: datasets + 5")
        ap.add_argument("--min-ram", type=float, help="GB; default: 100 for prepare_data")
    rest = argv[1:]
    split = rest.index("--") if "--" in rest else len(rest)  # after --: overrides verbatim
    a, overrides = ap.parse_known_args(rest[:split])
    if unknown := [o for o in overrides if o.startswith("--")]:
        ap.error(f"unrecognized arguments: {' '.join(unknown)}")
    overrides += rest[split + 1:]
    if command == "kaggle":
        return submit_kaggle(a, overrides)
    if command == "auto":
        return auto(a, overrides)
    (sweep_hf if a.sweep else submit_hf)(a, overrides)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
