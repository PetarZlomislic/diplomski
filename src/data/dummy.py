import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from src.data.base import BaseDataModule


class _DictTensorDataset(Dataset):
    def __init__(self, tensors: dict[str, Tensor]):
        self.tensors = tensors
        self.n = next(iter(tensors.values())).shape[0]

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, i: int) -> dict[str, Tensor]:
        return {k: v[i] for k, v in self.tensors.items()}


class DummyDataModule(BaseDataModule):
    """Deterministic random tensors from `seed`. No file I/O; ignores `data_uri`."""

    def __init__(
        self,
        num_classes: int,
        in_channels: dict[str, int],
        image_size: int,
        n_samples: int = 64,
        batch_size: int = 8,
        num_workers: int = 0,
        seed: int = 0,
        data_uri: str = "",
    ):
        super().__init__()
        self.num_classes = num_classes
        self.in_channels = dict(in_channels)
        self.image_size = image_size
        self.modalities = tuple(self.in_channels)
        self.n_samples = n_samples
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seed = seed
        self.data_uri = data_uri

    def _make_split(self, n: int, offset: int) -> _DictTensorDataset:
        g = torch.Generator().manual_seed(self.seed + offset)
        tensors = {
            m: torch.randn(n, c, self.image_size, self.image_size, generator=g)
            for m, c in self.in_channels.items()
        }
        tensors["label"] = torch.randint(0, 2, (n, self.num_classes), generator=g).float()
        return _DictTensorDataset(tensors)

    def setup(self, stage: str | None = None) -> None:
        n_val = n_test = max(1, self.n_samples // 5)
        n_train = max(1, self.n_samples - n_val - n_test)
        self.train_ds = self._make_split(n_train, 0)
        self.val_ds = self._make_split(n_val, 1)
        self.test_ds = self._make_split(n_test, 2)

    def _loader(self, ds: Dataset, shuffle: bool) -> DataLoader:
        return DataLoader(
            ds, batch_size=self.batch_size, shuffle=shuffle, num_workers=self.num_workers
        )

    def train_dataloader(self) -> DataLoader:
        return self._loader(self.train_ds, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return self._loader(self.val_ds, shuffle=False)

    def test_dataloader(self) -> DataLoader:
        return self._loader(self.test_ds, shuffle=False)
