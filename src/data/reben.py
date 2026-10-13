"""reBEN (BigEarthNet v2.0) source for src.prepare_data.

Reads the LMDB published by the reBEN authors on the Hub (hackelle/BigEarthNetV2-LMDB):
one SafeTensors record per patch, keyed by the S2 `patch_id` (bands B01..B12) or by the
S1 `s1_name` (VV, VH), plus the official `metadata.parquet`. The parquet is the recommended
set: patches with seasonal snow, cloud or cloud shadow are already excluded.

Patches are read LAZILY: a record holds `PatchRef`s, and pixels are only read (in the
litdata worker) when a shard is written. The 20 m and 60 m S2 bands are bilinearly
resampled to 120 x 120 here, once.

Sources:  hf://datasets/<repo>  (downloaded into `work_dir`)  or a local directory
          containing BENv2.lmdb/ and metadata.parquet.
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

DEFAULT_SOURCE = "hf://datasets/hackelle/BigEarthNetV2-LMDB"
LMDB_DIR = "BENv2.lmdb"
METADATA = "metadata.parquet"
SIZE = 120
S2_BANDS = ("B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B11", "B12")
S1_BANDS = ("VV", "VH")  # backscatter in dB, not speckle-filtered
SPLITS = {"train": "train", "validation": "val", "test": "test"}
# The 19-class nomenclature, in the order of the official description (Table 1).
CLASSES = (
    "Urban fabric",
    "Industrial or commercial units",
    "Arable land",
    "Permanent crops",
    "Pastures",
    "Complex cultivation patterns",
    "Land principally occupied by agriculture, with significant areas of natural vegetation",
    "Agro-forestry areas",
    "Broad-leaved forest",
    "Coniferous forest",
    "Mixed forest",
    "Natural grassland and sparsely vegetated areas",
    "Moors, heathland and sclerophyllous vegetation",
    "Transitional woodland, shrub",
    "Beaches, dunes, sands",
    "Inland wetlands",
    "Coastal wetlands",
    "Inland waters",
    "Marine waters",
)

_ENVS: dict[str, Any] = {}  # one read-only LMDB handle per process (litdata workers)


def _env(path: str):
    if path not in _ENVS:
        import lmdb

        _ENVS[path] = lmdb.open(path, readonly=True, lock=False, readahead=False,
                                max_readers=512, subdir=True)
    return _ENVS[path]


def _resize(x: np.ndarray, size: int) -> np.ndarray:
    import torch
    import torch.nn.functional as F

    t = torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32))[None, None]
    return F.interpolate(t, size=(size, size), mode="bilinear", align_corners=False)[0, 0].numpy()


@dataclass(frozen=True)
class PatchRef:
    """A not-yet-read patch: `load()` returns (len(bands), SIZE, SIZE).
    S2 stays uint16 (reflectance, half the bytes of float32); S1 is float32 dB."""

    lmdb_path: str
    key: str
    bands: tuple[str, ...]

    def load(self) -> np.ndarray:
        from safetensors.numpy import load as st_load

        with _env(self.lmdb_path).begin() as txn:
            raw = txn.get(self.key.encode())
        if raw is None:
            raise KeyError(f"{self.key} not in {self.lmdb_path}")
        tensors = st_load(raw)
        out = []
        for b in self.bands:
            x = tensors[b]
            if x.shape != (SIZE, SIZE):
                x = _resize(x, SIZE)
            out.append(np.asarray(x, dtype=np.float32))
        arr = np.stack(out)
        if self.bands == S1_BANDS:
            # dB of a zero backscatter is -inf; clamp non-finite values to a floor.
            return np.nan_to_num(arr, nan=-50.0, neginf=-50.0, posinf=30.0).astype(np.float32)
        return np.clip(np.rint(arr), 0, np.iinfo(np.uint16).max).astype(np.uint16)


def materialize(value: Any) -> Any:
    return value.load() if isinstance(value, PatchRef) else value


def multi_hot(labels) -> np.ndarray:
    idx = {c: i for i, c in enumerate(CLASSES)}
    unknown = set(labels) - set(idx)
    if unknown:
        raise ValueError(f"labels outside the 19-class nomenclature: {sorted(unknown)}")
    y = np.zeros(len(CLASSES), dtype=np.uint8)
    y[[idx[c] for c in labels]] = 1
    return y


def resolve(source: str, work_dir: Path) -> Path:
    """A local directory with BENv2.lmdb/ and metadata.parquet; hf://datasets/<repo> is
    downloaded into `work_dir` first (~155 GB: run this on a remote machine)."""
    if source.startswith("hf://datasets/"):
        from huggingface_hub import snapshot_download

        repo = source[len("hf://datasets/"):].strip("/")
        return Path(snapshot_download(repo, repo_type="dataset",
                                      local_dir=str(work_dir / repo.replace("/", "__"))))
    return Path(source)


def load_records(root: Path) -> list[dict[str, Any]]:
    import pandas as pd

    lmdb_path = str(root / LMDB_DIR)
    meta = pd.read_parquet(root / METADATA, columns=["patch_id", "labels", "split", "s1_name"])
    bad = set(meta["split"]) - set(SPLITS)
    if bad:
        raise ValueError(f"unexpected split values {bad}")
    return [
        {"patch_id": pid, "split": SPLITS[split], "label": multi_hot(labels),
         "optical": PatchRef(lmdb_path, pid, S2_BANDS),
         "sar": PatchRef(lmdb_path, s1, S1_BANDS)}
        for pid, labels, split, s1 in meta.itertuples(index=False)
    ]
