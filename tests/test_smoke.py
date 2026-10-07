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
