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
