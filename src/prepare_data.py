"""One-time data preparation: source -> per-tier, per-split litdata shards (-> Hub repos).

NOT part of the training path. Training streams the shards this writes.

Tiers keep all 14 bands (12 optical + 2 SAR); they subset PATCHES, never bands.

Sampling rules (both enforced):
  1. Sample within each official split, never across them (reBEN's splits are
     geographically disjoint; crossing them reintroduces spatial leakage).
  2. Label-aware: classes are processed rarest-first and topped up to `min_per_class`
     before the remaining budget is filled at random. Afterwards every class must have
     at least `min_per_class` patches in every split, or preparation fails.

Each tier's manifest.json also records per-band `band_stats` (mean/std over that tier's
train split), which the streaming datamodule standardises with.

Test shards additionally carry `sar_shift_<k>` for each `--temporal-steps` k: the SAR of
the patch k grid cells away in the same tile, looked up in the FULL source (see
src/degradations/temporal.py).

Usage:
  python -m src.prepare_data --source synthetic --fraction 0.01 --out-dir data/prepared
  python -m src.prepare_data --source synthetic --out-dir data/prepared --out-repo user/reben
  python -m src.prepare_data --source reben --out-repo user/reben      # remote only, see
                                                                        # scripts/prepare_hf.sh
"""
import argparse
import json
import os
import random
import tempfile
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np

SPLITS = ("train", "val", "test")
TIERS = {"dev": 0.01, "sweep": 0.15, "full": 1.0}
BANDS = {"optical": 12, "sar": 2}  # 14 bands total; never subset here
NUM_CLASSES = 19
TEMPORAL_STEPS = (1, 2, 4, 8)
SYNTHETIC_GRID = 10  # synthetic tiles are SYNTHETIC_GRID x SYNTHETIC_GRID patches

Record = dict[str, Any]


class ClassFloorError(ValueError):
    """A class falls below the per-class minimum in some split after sampling."""


def synthetic_source(
    n: int = 6000,
    num_classes: int = NUM_CLASSES,
    image_size: int = 8,
    seed: int = 0,
    starve_class: int | None = None,
) -> list[Record]:
    """Fake patches with a skewed multi-label distribution and fixed split assignment.

    Patch ids follow reBEN's `<tile>_<row>_<col>` layout on a SYNTHETIC_GRID^2 grid per
    tile, so the temporal-shift lookup is exercised. `starve_class` puts that class in only
    two train patches, so the per-class floor must fail for val/test. Used to prove the
    assertion fires.
    """
    rng = np.random.default_rng(seed)
    probs = 0.5 / np.arange(1, num_classes + 1) ** 0.8  # long-tailed, like land cover
    split_of = rng.choice(len(SPLITS), size=n, p=[0.6, 0.2, 0.2])
    records = []
    for i in range(n):
        label = (rng.random(num_classes) < probs).astype(np.uint8)
        if not label.any():
            label[rng.integers(num_classes)] = 1
        records.append({
            "patch_id": f"SYN_T{i // SYNTHETIC_GRID**2:04d}_{i // SYNTHETIC_GRID % SYNTHETIC_GRID}"
                        f"_{i % SYNTHETIC_GRID}",
            "split": SPLITS[split_of[i]],
            "label": label,
            **{m: rng.standard_normal((c, image_size, image_size)).astype(np.float32)
               for m, c in BANDS.items()},
        })
    if starve_class is not None:
        starved = 0
        for r in records:
            keep = r["split"] == "train" and starved < 2
            r["label"][starve_class] = int(keep)
            starved += keep
            if not r["label"].any():
                r["label"][0] = 1
    return records


def real_source(source: str, work_dir: Path) -> list[Record]:
    """reBEN records (see src/data/reben.py): same shape as `synthetic_source`, but the
    bands are lazy `PatchRef`s read only when shards are written. `source` is "reben"
    (the authors' LMDB on the Hub), hf://datasets/<repo>, or a local directory."""
    from src.data import reben

    root = reben.resolve(reben.DEFAULT_SOURCE if source == "reben" else source, work_dir)
    return reben.load_records(root)


def _array(value: Any) -> Any:
    """Pixels of a band: eager arrays pass through, lazy reBEN patches are read."""
    from src.data.reben import materialize

    return materialize(value)


def sample_split(
    records: list[Record], fraction: float, min_per_class: int, rng: random.Random
) -> list[Record]:
    """Greedy label-aware sampling within ONE split: rarest classes first up to the floor,
    then fill the remaining budget at random. The floor wins if it exceeds the budget."""
    budget = max(1, round(fraction * len(records)))
    if fraction >= 1.0:
        return list(records)
    num_classes = len(records[0]["label"])
    totals = np.sum([r["label"] for r in records], axis=0)
    chosen: set[int] = set()
    counts = np.zeros(num_classes, dtype=int)
    for k in np.argsort(totals, kind="stable"):  # rarest first
        need = min_per_class - counts[k]
        if need <= 0:
            continue
        candidates = [i for i, r in enumerate(records) if r["label"][k] and i not in chosen]
        for i in rng.sample(candidates, min(need, len(candidates))):
            chosen.add(i)
            counts += records[i]["label"]
    rest = [i for i in range(len(records)) if i not in chosen]
    chosen.update(rng.sample(rest, max(0, min(budget - len(chosen), len(rest)))))
    return [records[i] for i in sorted(chosen)]


def check_class_floor(by_split: dict[str, list[Record]], min_per_class: int) -> None:
    short = {}
    for split, recs in by_split.items():
        counts = np.sum([r["label"] for r in recs], axis=0) if recs else None
        missing = {} if counts is None else {
            int(k): int(c) for k, c in enumerate(counts) if c < min_per_class
        }
        if missing:
            short[split] = missing
    if short:
        raise ClassFloorError(f"classes below min_per_class={min_per_class}: {short}")


def subset(
    records: list[Record], fraction: float, min_per_class: int, seed: int = 0
) -> dict[str, list[Record]]:
    """Subset every split independently and enforce the per-class floor."""
    rng = random.Random(seed)
    by_split = {s: [r for r in records if r["split"] == s] for s in SPLITS}
    out = {s: sample_split(recs, fraction, min_per_class, rng) if recs else []
           for s, recs in by_split.items()}
    check_class_floor(out, min_per_class)
    return out


def grid_position(patch_id: str) -> tuple[str, int, int]:
    """`<tile>_<row>_<col>` -> (tile, row, col). The tile part is everything before the last
    two fields, so for reBEN it is the S2 product + tile: neighbours share an acquisition."""
    tile, row, col = patch_id.rsplit("_", 2)
    return tile, int(row), int(col)


def _ring(d: int) -> list[tuple[int, int]]:
    """Offsets at Chebyshev distance d: cardinal, then diagonal, then the rest, in order."""
    first = [(0, d), (d, 0), (0, -d), (-d, 0), (d, d), (d, -d), (-d, d), (-d, -d)]
    rest = sorted((r, c) for r in range(-d, d + 1) for c in range(-d, d + 1)
                  if max(abs(r), abs(c)) == d and (r, c) not in first)
    return first + rest


def shifted_neighbour(
    patch_id: str, step: int, grid: dict[str, dict[tuple[int, int], Record]],
    tiles: list[str],
) -> tuple[Record, str]:
    """The patch `step` cells away in the same tile, and how it was found:
    "exact" (distance == step), "farther" (nearest ring beyond), "nearer" (nearest ring
    below; still >= 1 cell away), or "cross_tile" (the tile has no other patch: first patch
    of the next tile). Deterministic: the same inputs always give the same neighbour."""
    tile, row, col = grid_position(patch_id)
    cells = grid[tile]
    span = max(max(abs(r - row), abs(c - col)) for r, c in cells)
    order = ([(step, "exact")] + [(d, "farther") for d in range(step + 1, span + 1)]
             + [(d, "nearer") for d in range(step - 1, 0, -1)])
    for d, how in order:
        for dr, dc in _ring(d):
            if (row + dr, col + dc) in cells:
                return cells[(row + dr, col + dc)], how
    other = tiles[(tiles.index(tile) + 1) % len(tiles)]
    if other == tile:
        raise ValueError("temporal shift needs at least two patches in the source")
    return grid[other][min(grid[other])], "cross_tile"


def attach_shifts(
    test: list[Record], source: list[Record], steps: Iterable[int], modality: str = "sar"
) -> tuple[list[Record], dict[str, dict[str, int]]]:
    """Copies of `test` with `<modality>_shift_<k>` added, neighbours looked up in `source`
    (all splits, unsubsampled). Returns the records and per-step counts of how each
    neighbour was found."""
    grid: dict[str, dict[tuple[int, int], Record]] = {}
    for r in source:
        tile, row, col = grid_position(r["patch_id"])
        grid.setdefault(tile, {})[(row, col)] = r
    tiles = sorted(grid)
    out = [dict(r) for r in test]
    counts: dict[str, dict[str, int]] = {}
    for k in steps:
        c = counts.setdefault(str(k), {"exact": 0, "farther": 0, "nearer": 0, "cross_tile": 0})
        for r in out:
            nb, how = shifted_neighbour(r["patch_id"], k, grid, tiles)
            r[f"{modality}_shift_{k}"] = nb[modality]
            c[how] += 1
    return out, counts


def band_stats(records: list[Record], max_records: int = 5000,
               seed: int = 0) -> dict[str, dict[str, list[float]]]:
    """Per-band mean and std over every pixel of `records` (float64 accumulation), from a
    fixed random sample of at most `max_records` patches."""
    if len(records) > max_records:
        records = random.Random(seed).sample(records, max_records)
    stats = {}
    for m in BANDS:
        n, s, ss = 0, 0.0, 0.0
        for r in records:
            x = np.asarray(_array(r[m]), dtype=np.float64)
            n += x.shape[1] * x.shape[2]
            s = s + x.sum(axis=(1, 2))
            ss = ss + (x * x).sum(axis=(1, 2))
        mean = s / n
        std = np.sqrt(np.maximum(ss / n - mean**2, 0.0))
        stats[m] = {"mean": [float(v) for v in mean],
                    "std": [float(v) if v > 0 else 1.0 for v in std]}
    return stats


def _to_sample(record: Record) -> Record:
    """litdata worker fn; module-level so it pickles under spawn (Windows)."""
    return {k: _array(v) for k, v in record.items() if k != "split"}


def write_shards(by_split: dict[str, list[Record]], out_dir: Path, chunk_bytes: str = "128MB",
                 num_workers: int = 1) -> dict[str, int]:
    from litdata import optimize

    sizes = {}
    for split, recs in by_split.items():
        if not recs:
            continue
        optimize(fn=_to_sample, inputs=recs, output_dir=str(out_dir / split),
                 chunk_bytes=chunk_bytes, num_workers=num_workers, mode="overwrite",
                 verbose=False)
        sizes[split] = len(recs)
    return sizes


def push_tier(out_dir: Path, repo_id: str, token: str | None = None) -> None:
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    api.create_repo(repo_id, repo_type="dataset", exist_ok=True, private=True)
    # One commit per split, then the manifest last: a failure loses at most one split, and
    # a tier without manifest.json is visibly incomplete.
    for split_dir in sorted(p for p in out_dir.iterdir() if p.is_dir()):
        api.upload_folder(folder_path=str(split_dir), path_in_repo=split_dir.name,
                          repo_id=repo_id, repo_type="dataset",
                          commit_message=f"shards: {out_dir.name}/{split_dir.name}")
        _event("split_pushed", repo=repo_id, split=split_dir.name)
    api.upload_file(path_or_fileobj=str(out_dir / "manifest.json"),
                    path_in_repo="manifest.json", repo_id=repo_id, repo_type="dataset",
                    commit_message=f"manifest: {out_dir.name}")


def _event(event: str, **fields: Any) -> None:
    print(json.dumps({"event": event, **fields}), flush=True)


def tier_name(fraction: float) -> str:
    return next((k for k, v in TIERS.items() if v == fraction), f"frac{fraction:g}")


def prepare(
    records: list[Record],
    out_dir: Path,
    fractions: Iterable[float],
    min_per_class: int = 1,
    seed: int = 0,
    out_repo: str | None = None,
    token: str | None = None,
    temporal_steps: Iterable[int] = TEMPORAL_STEPS,
    num_workers: int = 1,
    stats_samples: int = 5000,
) -> dict[str, dict[str, int]]:
    """Write each tier. Temporal neighbours come from the FULL source but the SAME split,
    so a test patch is never paired with a training patch's SAR."""
    temporal_steps = list(temporal_steps)
    manifest = {}
    for fraction in fractions:
        name = tier_name(fraction)
        by_split = subset(records, fraction, min_per_class, seed)
        shifts = {}
        if temporal_steps and by_split["test"]:
            pool = [r for r in records if r["split"] == "test"]
            by_split["test"], shifts = attach_shifts(by_split["test"], pool, temporal_steps)
        tier_dir = out_dir / name
        sizes = write_shards(by_split, tier_dir, num_workers=num_workers)
        class_counts = {
            s: [int(c) for c in np.sum([r["label"] for r in recs], axis=0)]
            for s, recs in by_split.items() if recs
        }
        (tier_dir / "manifest.json").write_text(json.dumps({
            "fraction": fraction, "min_per_class": min_per_class, "seed": seed,
            "bands": BANDS, "sizes": sizes, "class_counts": class_counts,
            "band_stats": (band_stats(by_split["train"], stats_samples, seed)
                           if by_split["train"] else None),
            "temporal": {"steps": temporal_steps, "neighbour_found": shifts},
        }, indent=2))
        _event("tier_written", tier=name, sizes=sizes)
        if out_repo:
            push_tier(tier_dir, f"{out_repo}-{name}", token)
            _event("tier_pushed", tier=name, repo=f"{out_repo}-{name}")
        manifest[name] = sizes
    return manifest


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--source", required=True,
                    help='"synthetic", "reben" (the authors\' LMDB on the Hub), '
                         "hf://datasets/<repo>, or a local dir with BENv2.lmdb + metadata.parquet")
    ap.add_argument("--out-dir", type=Path, default=Path("data/prepared"))
    ap.add_argument("--out-repo", default=None,
                    help="HF dataset repo prefix; each tier goes to <prefix>-<tier>")
    ap.add_argument("--fraction", type=float, default=None,
                    help="single tier at this fraction (default: dev, sweep and full)")
    ap.add_argument("--min-per-class", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--split-aware", action=argparse.BooleanOptionalAction, default=True,
                    help="sample within official splits (cannot be disabled; kept explicit)")
    ap.add_argument("--temporal-steps", type=int, nargs="*", default=list(TEMPORAL_STEPS),
                    help="grid steps of shifted SAR to store in test shards (none: disable)")
    ap.add_argument("--synthetic-n", type=int, default=6000)
    ap.add_argument("--work-dir", type=Path, default=Path(tempfile.gettempdir()) / "reben",
                    help="where a Hub source is downloaded (needs ~155 GB)")
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 1,
                    help="shard-writing processes")
    ap.add_argument("--stats-samples", type=int, default=5000,
                    help="train patches sampled for band_stats")
    args = ap.parse_args()
    if not args.split_aware:
        ap.error("--no-split-aware is refused: sampling across splits leaks test geography")

    from src.env import detect

    records = (synthetic_source(args.synthetic_n, seed=args.seed) if args.source == "synthetic"
               else real_source(args.source, args.work_dir))
    _event("source_loaded", source=args.source, records=len(records))
    fractions = [args.fraction] if args.fraction is not None else list(TIERS.values())
    manifest = prepare(records, args.out_dir, fractions, args.min_per_class, args.seed,
                       args.out_repo, detect().secret("HF_TOKEN"), args.temporal_steps,
                       args.workers, args.stats_samples)
    print(json.dumps({"event": "prepare_done", "out_dir": str(args.out_dir),
                      "tiers": manifest}))


if __name__ == "__main__":
    main()
