import csv
from dataclasses import asdict, dataclass, fields
from pathlib import Path


@dataclass
class ResultsRow:
    run_id: str
    git_sha: str
    config_hash: str        # hash of the resolved training config
    timestamp: str
    model: str
    data: str
    degradation: str
    severity: float
    seed: int
    split: str              # degradation sweeps run on "test"
    ckpt_kind: str          # "best" | "last" | explicit path
    threshold: float        # multi-label decision threshold used
    f1_macro: float
    f1_micro: float
    accuracy: float
    ece: float | None
    params: int
    latency_ms: float | None


FIELDNAMES = [f.name for f in fields(ResultsRow)]
RUN_CONFIG = "run_config.yaml"
RESULTS_DIR = "results"  # <run dir>/results/<eval_id>.csv, one file per evaluation


def append_rows(rows: list[ResultsRow], path: Path) -> None:
    """Append to one CSV per run; aggregate.py merges them. Writes the header if absent."""
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists()
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if new:
            writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def push_rows(csv_path: Path, repo_id: str, token: str | None) -> bool:
    """Upload one evaluation's CSV to `results/<run_id>/<eval_id>.csv` in a Hub dataset repo.
    Never raises: a failed push must not kill the run, since the local file is still written."""
    import logging

    run_id = csv_path.parent.parent.name
    try:
        from huggingface_hub import HfApi

        HfApi(token=token).upload_file(
            path_or_fileobj=str(csv_path),
            path_in_repo=f"results/{run_id}/{csv_path.name}",
            repo_id=repo_id,
            repo_type="dataset",
            commit_message=f"results for {run_id}/{csv_path.stem}",
        )
        return True
    except Exception as e:  # network, auth, missing repo, ...
        logging.getLogger(__name__).warning("results push to %s failed: %s", repo_id, e)
        return False
