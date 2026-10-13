import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from watch_jobs import JobState, apply_event, jobs_from_args, parse_line, render  # noqa: E402


def test_events_fold_into_job_state() -> None:
    st = JobState("o/j", "opt s0")
    lines = [
        '{"ts": "1", "event": "dataset_download_done", "gb": 41.5, "seconds": 97}',
        'noise {"ts": "2", "event": "heartbeat", "phase": "train", "epoch": 3, "step": 10, '
        '"max_steps": 20, "samples_per_s": 1300.0, "data_wait_frac": 0.4}',
        '{"ts": "3", "event": "run_end", "best_val_f1_macro": 0.57}',
        '{"ts": "4", "event": "eval_progress", "degradation": "none", "f1_macro": 0.55}',
        "Downloading bytes: 50%",
    ]
    for line in lines:
        if (e := parse_line(line)) is not None:
            apply_event(st, e)
    assert (st.epoch, st.step, st.max_steps, st.data_wait) == (3, 10, 20, 0.4)
    assert st.val_f1 == 0.57 and st.test_f1 == 0.55 and "41.5 GB" in st.note
    out = render([st], wait_limit=0.3, started=0.0)
    assert "0.40!" in out and "opt s0" in out


def test_jobs_from_sweep_tsv(tmp_path: Path) -> None:
    tsv = tmp_path / "s.tsv"
    tsv.write_text("job_id\toverrides\no/1\t+experiment=x model=late seed=2\n")
    assert jobs_from_args([str(tsv)]) == [("o/1", "late s2")]
