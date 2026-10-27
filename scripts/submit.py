"""Run entrypoints remotely at the current commit, and watch them.

  python scripts/submit.py hf [--dry-run] [--allow-dirty] [--flavor F] [--timeout T]
                              [--entry E] [--data owner/dataset]... [--sweep] <overrides...>
  python scripts/submit.py kaggle [--dry-run] [--allow-dirty] [--entry E] [--kernel user/slug]
                                  [--data owner/dataset]... <overrides...>
  uv run python scripts/submit.py auto [--dry-run] [--time-value USD_PER_H] [--deadline T]
                                [--sweep] [--entry E] [--data owner/dataset]... [--[no-]gpu]
                                [--min-disk GB] [--min-ram GB] [--timeout T] <overrides...>
  python scripts/submit.py watch [<sweep.tsv> | <owner>/<job_id>...] [--once]

auto    predict the job's wall time and cost on Kaggle and every HF Jobs flavor (scripts/plan.py,
        calibrated on real pilot jobs), drop what cannot run it right now (GPU, disk, RAM,
        Kaggle's idle kernel / session cap / weekly GPU quota, --deadline), and submit to the
        lowest cost + time_value x hours (--time-value, $/h, default 1: 0 = cheapest, high =
        fastest). Prints every option; --timeout defaults to 1.5x the prediction + 15 min.

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
(forwarded when set), KAGGLE_KERNEL, REPO_SLUG (default: from the origin remote), and
KAGGLE_MIRRORS="<hub repo>=<kaggle dataset> ...": private Kaggle copies of Hub datasets that
a Kaggle kernel mounts instead of downloading (no HF token needed in the kernel).
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


class NoCredit(Exception):
    """HF Jobs refused the job: the account's pre-paid credit is used up (402)."""


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


def kaggle_mirrors() -> dict[str, str]:
    """KAGGLE_MIRRORS="<hub repo>=<kaggle dataset> ...": private Kaggle copies of Hub datasets.
    Attached to the kernel instead of downloaded, so the kernel needs no HF token."""
    return dict(m.split("=", 1) for m in os.environ.get("KAGGLE_MIRRORS", "").split())


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
        if "402" in res.stderr or "credit balance is insufficient" in res.stderr:
            raise NoCredit(res.stderr)
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
    repos, mirrors = datasets(overrides, a.data).split(), kaggle_mirrors()
    mirrored = {r: mirrors[r] for r in repos if r in mirrors}
    stamp = {"REPO_SLUG": slug, "REPO_SHA": sha, "ENTRY": a.entry,
             "HYDRA_OVERRIDES": shlex.join(overrides),  # split again by src/env.py
             "DATASETS": " ".join(r for r in repos if r not in mirrored),
             "KAGGLE_MIRRORS": " ".join(f"{r}={k}" for r, k in mirrored.items()),
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
            "dataset_sources": sorted(set(mirrored.values())),
            "competition_sources": [], "kernel_sources": [], "model_sources": []}
    (build / "kernel-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    out_dir = ROOT / "outputs" / "kaggle" / safe / sha
    push = ["kaggle", "kernels", "push", "-p", str(build)]
    if a.dry_run:
        print(f"# kernel {kernel} at {slug}@{sha}, RUN_ENV=kaggle "
              f"HYDRA_OVERRIDES={stamp['HYDRA_OVERRIDES']} DATASETS={stamp['DATASETS']}"
              + (f" KAGGLE_MIRRORS={stamp['KAGGLE_MIRRORS']}" if mirrored else ""))
        for c in (push, ["kaggle", "kernels", "status", kernel],
                  ["kaggle", "kernels", "output", kernel, "-p", str(out_dir)]):
            print(shlex.join(c))
        return 0
    if not shutil.which("kaggle"):
        die("kaggle CLI not found (uv tool install kaggle)")
    os.environ["PYTHONUTF8"] = "1"  # the Kaggle CLI crashes printing non-ASCII on Windows
    subprocess.run(push, check=True)
    print(f"pushed {kernel}; polling every 30s")
    status = ""
    while True:
        try:  # a call that never answers must not freeze the wait (it once hung for 2 h)
            res = subprocess.run(["kaggle", "kernels", "status", kernel], capture_output=True,
                                 text=True, timeout=120)
        except subprocess.TimeoutExpired:
            res = subprocess.CompletedProcess([], 1, "", "status call timed out after 120 s")
        m = re.search(r"KernelWorkerStatus\.(\w+)", res.stdout)
        now = time.strftime("%H:%M:%S", time.gmtime())
        if not m:  # network blip or CLI error: the kernel keeps running, so keep polling
            err = (res.stderr or res.stdout).strip()[-120:]
            print(f"{now} status unavailable, retrying: {err}")
        elif m[1] != status:
            status = m[1]
            print(f"{now} {kernel}: {status}")
        if status in ("COMPLETE", "ERROR", "CANCEL_REQUESTED", "CANCEL_ACKNOWLEDGED"):
            break
        time.sleep(30)
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "kernel_log.json").open("w") as f:
        subprocess.run(["kaggle", "-W", "kernels", "logs", kernel], stdout=f,
                       stderr=subprocess.STDOUT, timeout=600)
    download_outputs(kernel, out_dir)
    print(f"outputs: {out_dir} (execution log: {out_dir / 'kernel_log.json'})")
    return 0 if status == "COMPLETE" else 1


def download_outputs(kernel: str, out_dir: Path) -> int:
    """Every output file of the kernel's latest version. The CLI pages its results (20 files
    by default), so follow the page tokens; returns the number of files downloaded."""
    token, n = None, 0
    while True:
        cmd = ["kaggle", "-W", "kernels", "output", kernel, "-p", str(out_dir),
               "--page-size", "200", *(["--page-token", token] if token else [])]
        try:  # `kernels output` can hang when a failed kernel produced no files; bound it
            res = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
        except subprocess.TimeoutExpired:
            print("warning: output download timed out", file=sys.stderr)
            return n
        n += res.stdout.count("downloaded to")
        m = re.search(r"Next Page Token = (\S+)", res.stdout)
        if not m:
            return n
        token = m[1]


# ---------------------------------------------------------------- auto: speed vs price

JOB_OVERHEAD_GB = 5  # repo, uv env and caches next to the downloaded datasets


def kaggle_unusable(a: argparse.Namespace, w, hours: float) -> str | None:
    """Why the Kaggle kernel cannot take this job right now (None: it can)."""
    import plan

    if a.entry == "prepare_data":
        return "data preparation needs a big CPU machine"
    if len(w.jobs) > 1:
        return f"one kernel cannot run {len(w.jobs)} jobs at once (use --entry exp1_run)"
    if not shutil.which("kaggle"):
        return "kaggle CLI not found"
    kernel = a.kernel or os.environ.get("KAGGLE_KERNEL")
    if not kernel:
        return "KAGGLE_KERNEL not set"
    cap = plan.KAGGLE["gpu_h" if w.gpu else "cpu_h"]
    if hours * plan.SAFETY > cap:
        return f"~{plan.fmt_h(hours)} is too close to the {cap} h session cap"
    env = os.environ | {"PYTHONUTF8": "1"}
    status = subprocess.run(["kaggle", "kernels", "status", kernel], capture_output=True,
                            text=True, env=env).stdout
    if re.search(r"running|queued", status, re.IGNORECASE):
        return f"{kernel} is still running (a push would replace it)"
    if w.gpu:
        gpu, err = None, ""
        for attempt in range(3):  # transient network errors happen; retry before giving up
            res = subprocess.run(["kaggle", "quota", "--format", "json"], capture_output=True,
                                 text=True, env=env)
            try:  # the CLI may print an "outdated version" warning line before the JSON
                quota = json.loads(res.stdout[res.stdout.index("["):])
                gpu = next(q for q in quota if q["resource"] == "GPU")
                break
            except (ValueError, StopIteration):
                lines = [ln for ln in (res.stderr + res.stdout).splitlines()
                         if ln.strip() and "outdated" not in ln]
                err = (lines or [""])[-1][:120]
                time.sleep(5 * (attempt + 1))
        if gpu is None:
            return f"cannot read the GPU quota (`kaggle quota`: {err or 'no JSON'})"
        left = float(gpu["remaining"].rstrip("h"))
        if left < hours * plan.SAFETY:
            return f"GPU quota {left:g} h left (refills {gpu['refreshAt']})"
    return None


def hf_unusable(api) -> str | None:
    if not os.environ.get("HF_BUCKET"):
        return "HF_BUCKET not set"
    try:
        me = api.whoami()
    except Exception as e:  # network blip or rate limit: skip HF rather than crash
        return f"cannot reach the Hub ({type(e).__name__})"
    if not (me.get("isPro") or me.get("canPay") or me.get("orgs")):
        return "account cannot run Jobs (needs PRO or billing)"
    return None


def auto(a: argparse.Namespace, overrides: list[str]) -> int:
    """Predict time and cost everywhere usable now; submit to the best trade-off."""
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")  # manifest download
    import plan
    from huggingface_hub import HfApi

    api = HfApi(token=os.environ.get("HF_TOKEN"))
    combos = expand(overrides) if a.sweep else [overrides]
    repos = sorted({r for c in combos for r in datasets(c, a.data).split()})
    infos = {r: api.dataset_info(r, files_metadata=True) for r in repos}
    gb = {r: sum((s.size or 0) for s in i.siblings) / 1e9 for r, i in infos.items()}
    data_gb = sum(gb.values())
    unmirrored = [r for r in repos if r not in kaggle_mirrors()]  # Kaggle must download these
    private = [r for r in unmirrored if infos[r].private]
    w = plan.Work([plan.job_runs(a.entry, c, repos, api) for c in combos], data_gb)
    if a.gpu is not None:
        for r in (r for j in w.jobs if j for r in j):
            r.gpu = a.gpu
    prep = a.entry == "prepare_data"
    need_disk = a.min_disk or (500 if prep else data_gb + JOB_OVERHEAD_GB)
    need_ram = a.min_ram or (100 if prep else 0)
    timeout_h = seconds(default_timeout(a)) / 3600

    kaggle = plan.kaggle_option()
    kaggle.download_gb = sum(gb[r] for r in unmirrored)
    if not prep:
        kaggle.need_disk = kaggle.download_gb + JOB_OVERHEAD_GB
    plan.predict(w, kaggle, timeout_h)
    kaggle.why_not = kaggle_unusable(a, w, kaggle.hours)
    hf = plan.hf_options(api)
    hf_why = hf_unusable(api)  # once: whoami is rate-limited
    for o in hf:
        o.why_not = hf_why
    deadline_h = a.deadline and seconds(a.deadline) / 3600
    timeout_given = a.timeout
    while True:
        ranked = plan.plan(w, [kaggle, *hf], a.time_value, deadline_h, timeout_h, need_disk,
                           need_ram)
        show_plan(ranked, a, w, data_gb)
        best = ranked[0]
        if best.why_not:
            die("nothing can run this job right now")
        a.min_free_gb = (round(need_disk) if best.name != "kaggle" and repos else
                         round(kaggle.need_disk) if unmirrored else None)
        a.gpu = w.gpu
        a.timeout = timeout_given or (plan.suggested_timeout(best.hours) if w.modelled
                                      else None)
        if best.name == "kaggle":
            if private:  # Kaggle cannot say whether a secret is attached, so only warn
                print(f"note: {', '.join(private)} is private: the kernel needs the Kaggle "
                      "Secret HF_TOKEN attached (kernel page -> Add-ons -> Secrets)")
            return submit_kaggle(a, overrides)
        a.flavor = best.name
        try:
            (sweep_hf if a.sweep else submit_hf)(a, overrides)
            return 0
        except NoCredit:  # the API does not expose the balance, so this is how we learn it
            print("HF Jobs: pre-paid credit used up (402); re-planning without HF Jobs")
            for o in hf:
                o.why_not = "no HF Jobs credit (402 Payment Required)"


def show_plan(ranked, a: argparse.Namespace, w, data_gb: float) -> None:
    import plan

    runs = [r for j in w.jobs if j for r in j]
    print(f"job: {len(w.jobs)} job(s), {len(runs)} run(s), "
          f"{sum(r.train for r in runs) / 1e3:.0f}k training samples, "
          f"{'GPU' if w.gpu else 'CPU'}, data {data_gb:.1f} GB; "
          f"an hour of waiting is worth ${a.time_value:g}"
          + ("" if w.modelled else "; no speed model for this entry: priced at --timeout"))
    print(f"     {'option':<16}{'time':>9}{'cost':>9}{'score':>9}")
    for o in ranked:
        if w.gpu and o.gpu is None:
            continue  # a GPU job: CPU flavors are not candidates
        if o.why_not:
            print(f"     {o.name:<16}{'':>27}  {o.why_not}")
            continue
        spec = f"{o.gpu or 'CPU'}, {o.vcpu} vCPU" + (", input-bound" if o.loader_bound else "")
        tags = f"  [{', '.join(o.tags)}]" if o.tags else ""
        cost, score = f"${o.cost:.2f}", f"${o.cost + a.time_value * o.hours:.2f}"
        print(f"  {'->' if 'pick' in o.tags else '  '} {o.name:<16}{plan.fmt_h(o.hours):>9}"
              f"{cost:>9}{score:>9}  {spec}{tags}")
    best = ranked[0]
    room = best.vcpu - plan.WORKER_RESERVED_VCPU
    few = [r.workers for r in runs if r.workers < room]
    if not best.why_not and best.loader_bound and few:
        print(f"hint: {best.name} has {best.vcpu} vCPUs but runs use data.num_workers={min(few)}; "
              f"data.num_workers={room} would load faster (it may change the sample order, so "
              "keep it identical across arms)")


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
    sys.stdout.reconfigure(line_buffering=True)  # keep our lines in order with the CLIs'
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
        ap.add_argument("--time-value", type=float, default=1.0, metavar="USD_PER_H",
                        help="what an hour of waiting is worth: 0 = cheapest, high = fastest")
        ap.add_argument("--deadline", help="must finish within, e.g. 2h or 90m")
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
