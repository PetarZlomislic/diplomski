"""Live terminal dashboard for HF Jobs launched by this repo. Runs locally, polls the Hub.

  uv run python scripts/watch_jobs.py                       # newest outputs/sweeps/*.tsv
  uv run python scripts/watch_jobs.py outputs/sweeps/<ts>.tsv
  uv run python scripts/watch_jobs.py <owner>/<job_id> ...  # explicit jobs
  uv run python scripts/watch_jobs.py --once                # print one table and exit

Each poll inspects every job and reads the tail of its log, parsing the JSON events our
entrypoints emit (heartbeat, run_start/run_end/run_failed, eval_progress, dataset_download_*,
prepare_data events). State is merged across polls, so values survive the log tail moving on.
Exits by itself once every job has finished. HF_TOKEN comes from the environment or .env.
"""
import argparse
import concurrent.futures as cf
import json
import os
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
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


def load_env() -> None:
    env = ROOT / ".env"
    if not env.exists():
        return
    for line in env.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip().removeprefix("export ").strip(),
                                  v.strip().strip("'\""))


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
        if not line.strip():
            continue
        job_id, overrides = (line.split("\t") + [""])[:2]
        m = re.search(r"model=(\S+)", overrides)
        s = re.search(r"seed=(\S+)", overrides)
        label = (m.group(1) if m else job_id[-8:]) + (f" s{s.group(1)}" if s else "")
        out.append((job_id, label))
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
        st.phase = e.get("phase", st.phase) or st.phase
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
    if i < 0:
        return None
    try:
        e = json.loads(line[i:])
    except json.JSONDecodeError:
        return None
    return e if isinstance(e, dict) and "event" in e else None


def poll(api, st: JobState, tail: int) -> None:
    owner, _, jid = st.job_id.rpartition("/")
    info = api.inspect_job(job_id=jid, namespace=owner or None)
    st.stage = info.status.stage if info.status else "?"
    st.flavor = info.flavor or ""
    st.running_s = float(getattr(info.durations, "running_secs", None) or 0)
    if st.stage in ("SCHEDULING",):
        return
    for line in api.fetch_job_logs(job_id=jid, namespace=owner or None, tail=tail):
        if "Traceback" in line and not st.note.startswith("FAILED"):
            st.note = "Traceback in log"
        e = parse_line(line)
        if e is None:
            continue
        key = (e.get("ts"), e.get("event"), e.get("step"), e.get("split"))
        if key in st.seen:
            continue
        st.seen.add(key)
        apply_event(st, e)
    if info.status and info.status.message and st.stage == "ERROR":
        st.note = st.note or info.status.message[:60]


def fmt(v, spec: str = "", none: str = "-") -> str:
    return none if v is None else format(v, spec)


def render(states: list[JobState], wait_limit: float, started: float) -> str:
    hdr = (f"{'job':<16}{'stage':<11}{'phase':<10}{'ep':>4}{'step':>12}{'smp/s':>8}"
           f"{'wait':>7}{'val F1':>8}{'test F1':>8}{'run':>8}{'$':>7}  note")
    rows = [hdr, "-" * len(hdr)]
    total = 0.0
    for s in states:
        cost = s.running_s / 3600 * PRICE_PER_H.get(s.flavor, 0)
        total += cost
        step = f"{s.step}/{s.max_steps}" if s.step is not None else "-"
        wait = fmt(s.data_wait, ".2f")
        if s.data_wait is not None and s.data_wait >= wait_limit:
            wait += "!"
        rows.append(f"{s.label[:15]:<16}{s.stage[:10]:<11}{s.phase[:9]:<10}"
                    f"{fmt(s.epoch):>4}{step:>12}{fmt(s.samples_per_s, '.0f'):>8}{wait:>7}"
                    f"{fmt(s.val_f1, '.3f'):>8}{fmt(s.test_f1, '.3f'):>8}"
                    f"{int(s.running_s // 60):>5} m{cost:>7.2f}  {s.note}")
    done = sum(s.stage in TERMINAL for s in states)
    rows += ["", f"{done}/{len(states)} finished   est. cost ${total:.2f}   "
             f"wait! = data_wait >= {wait_limit}   updated {time.strftime('%H:%M:%S')}   "
             f"watching {int((time.time() - started) // 60)} m   Ctrl+C to quit"]
    return "\n".join(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("jobs", nargs="*", help="sweep TSV or <owner>/<job_id> ids")
    ap.add_argument("--interval", type=float, default=20, help="seconds between polls")
    ap.add_argument("--tail", type=int, default=120, help="log lines read per poll")
    ap.add_argument("--wait-limit", type=float, default=0.3)
    ap.add_argument("--once", action="store_true")
    args = ap.parse_args()

    load_env()
    from huggingface_hub import HfApi

    api = HfApi(token=os.environ.get("HF_TOKEN"))
    states = [JobState(j, label) for j, label in jobs_from_args(args.jobs)]
    os.system("")  # enables ANSI escapes in the Windows console
    started = time.time()
    pool = cf.ThreadPoolExecutor(max_workers=max(1, len(states)))
    pending: dict[str, cf.Future] = {}

    def safe_poll(s: JobState) -> None:
        try:
            poll(api, s, args.tail)
        except Exception as e:  # network blips must not kill the dashboard
            s.note = f"poll error: {type(e).__name__}"

    try:
        while True:
            for s in states:
                busy = s.job_id in pending and not pending[s.job_id].done()
                if busy or (s.stage in TERMINAL and s.seen):
                    continue
                pending[s.job_id] = pool.submit(safe_poll, s)
            # Wait for this round of polls, but never longer than one interval: a slow log
            # read just lands on a later refresh.
            cf.wait(list(pending.values()), timeout=args.interval)
            table = render(states, args.wait_limit, started)
            if args.once:
                print(table)
                return
            sys.stdout.write("\033[H\033[2J" + table + "\n")
            sys.stdout.flush()
            if all(s.stage in TERMINAL for s in states) and all(
                    f.done() for f in pending.values()):
                return
            time.sleep(args.interval)
    except KeyboardInterrupt:
        pass
    finally:
        pool.shutdown(wait=False, cancel_futures=True)


if __name__ == "__main__":
    main()
