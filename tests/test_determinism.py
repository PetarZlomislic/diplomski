import lightning as L
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

from src.evaluate import degraded_batches
from src.module import LitModule
from tests.conftest import compose_cfg


def _degraded(seed: int, degradation_seed: int, severity: float = 0.8) -> list[torch.Tensor]:
    cfg = compose_cfg(f"seed={seed}", "degradation=gaussian_noise", "+experiment=smoke")
    L.seed_everything(cfg.seed, workers=True)
    dm = instantiate(cfg.data)
    dm.setup("test")
    deg = instantiate(cfg.degradation)
    return [b["optical"] for b in degraded_batches(dm, deg, severity, degradation_seed)]


def _first_batch_loss(seed: int) -> torch.Tensor:
    cfg = compose_cfg(f"seed={seed}", "+experiment=smoke")
    L.seed_everything(cfg.seed, workers=True)
    dm = instantiate(cfg.data)
    dm.setup("fit")
    module = LitModule(
        model_cfg=OmegaConf.to_container(cfg.model, resolve=True),
        optim_cfg=OmegaConf.to_container(cfg.optim, resolve=True),
        threshold=cfg.threshold,
    )
    batch = next(iter(dm.train_dataloader()))
    return module.loss_fn(module(batch), batch["label"])


def test_same_seed_identical_first_batch_loss() -> None:
    assert torch.equal(_first_batch_loss(0), _first_batch_loss(0))


def test_same_degradation_seed_identical_degraded_tensors() -> None:
    a, b = _degraded(0, 12345), _degraded(0, 12345)
    assert all(torch.equal(x, y) for x, y in zip(a, b, strict=True))


def test_different_training_seed_same_degradation_seed_identical_tensors() -> None:
    """The invariant every cross-model comparison rests on."""
    a, b = _degraded(seed=0, degradation_seed=12345), _degraded(seed=1, degradation_seed=12345)
    assert all(torch.equal(x, y) for x, y in zip(a, b, strict=True))


def test_different_degradation_seed_changes_tensors() -> None:
    a, b = _degraded(0, 12345), _degraded(0, 999)
    assert not all(torch.equal(x, y) for x, y in zip(a, b, strict=True))
