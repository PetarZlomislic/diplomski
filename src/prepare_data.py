"""One-time data preparation: source -> per-tier, per-split litdata shards (-> Hub repos).

NOT part of the training path. Training streams the shards this writes.

Tiers keep all 14 bands (12 optical + 2 SAR); they subset PATCHES, never bands.

Sampling rules (both enforced):
  1. Sample within each official split, never across them (reBEN's splits are
     geographically disjoint; crossing them reintroduces spatial leakage).
  2. Label-aware: classes are processed rarest-first and topped up to `min_per_class`
     before the remaining budget is filled at random. Afterwards every class must have
     at least `min_per_class` patches in every split, or preparation fails.

Usage:
  python -m src.prepare_data --source synthetic --fraction 0.01 --out-dir data/prepared
  python -m src.prepare_data --source synthetic --out-dir data/prepared --out-repo user/reben
"""
import argparse
import json
import random
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np

SPLITS = ("train", "val", "test")
TIERS = {"dev": 0.01, "sweep": 0.15, "full": 1.0}
BANDS = {"optical": 12, "sar": 2}  # 14 bands total; never subset here
NUM_CLASSES = 19

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

    `starve_class` puts that class in only two train patches, so the per-class floor must
    fail for val/test. Used to prove the assertion fires.
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
            "patch_id": f"syn_{i:07d}",
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


def real_source(source: str) -> list[Record]:
    """TODO: reBEN loader (BigEarthNet v2, Zenodo record 10891137, ~118 GB).

    Must yield the same Record shape as `synthetic_source`: patch_id, the OFFICIAL split
    from reBEN's metadata parquet, a 19-dim multi-hot label, and all 12 optical + 2 SAR
    bands at a common resolution. Read patches lazily (paths, not arrays) at this scale.
    `source` is a local extracted directory or a list of Zenodo URLs.
    """
    raise NotImplementedError(f"real source not implemented yet: {source}")


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


def _to_sample(record: Record) -> Record:
    """litdata worker fn; module-level so it pickles under spawn (Windows)."""
    return {"patch_id": record["patch_id"], "label": record["label"],
            **{m: record[m] for m in BANDS}}


def write_shards(by_split: dict[str, list[Record]], out_dir: Path, chunk_bytes: str = "64MB",
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
    api.upload_folder(folder_path=str(out_dir), repo_id=repo_id, repo_type="dataset",
                      commit_message=f"shards: {out_dir.name}")


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
) -> dict[str, dict[str, int]]:
    manifest = {}
    for fraction in fractions:
        name = tier_name(fraction)
        by_split = subset(records, fraction, min_per_class, seed)
        tier_dir = out_dir / name
        sizes = write_shards(by_split, tier_dir)
        class_counts = {
            s: [int(c) for c in np.sum([r["label"] for r in recs], axis=0)]
            for s, recs in by_split.items() if recs
        }
        (tier_dir / "manifest.json").write_text(json.dumps({
            "fraction": fraction, "min_per_class": min_per_class, "seed": seed,
            "bands": BANDS, "sizes": sizes, "class_counts": class_counts,
        }, indent=2))
        if out_repo:
            push_tier(tier_dir, f"{out_repo}-{name}", token)
        manifest[name] = sizes
    return manifest


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--source", required=True, help='"synthetic", a local path, or Zenodo URLs')
    ap.add_argument("--out-dir", type=Path, default=Path("data/prepared"))
    ap.add_argument("--out-repo", default=None,
                    help="HF dataset repo prefix; each tier goes to <prefix>-<tier>")
    ap.add_argument("--fraction", type=float, default=None,
                    help="single tier at this fraction (default: dev, sweep and full)")
    ap.add_argument("--min-per-class", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--split-aware", action=argparse.BooleanOptionalAction, default=True,
                    help="sample within official splits (cannot be disabled; kept explicit)")
    ap.add_argument("--synthetic-n", type=int, default=6000)
    args = ap.parse_args()
    if not args.split_aware:
        ap.error("--no-split-aware is refused: sampling across splits leaks test geography")

    from src.env import detect

    records = (synthetic_source(args.synthetic_n, seed=args.seed) if args.source == "synthetic"
               else real_source(args.source))
    fractions = [args.fraction] if args.fraction is not None else list(TIERS.values())
    manifest = prepare(records, args.out_dir, fractions, args.min_per_class, args.seed,
                       args.out_repo, detect().secret("HF_TOKEN"))
    print(json.dumps({"event": "prepare_done", "out_dir": str(args.out_dir),
                      "tiers": manifest}))


if __name__ == "__main__":
    main()
