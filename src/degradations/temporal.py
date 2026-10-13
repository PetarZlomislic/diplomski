import torch
from torch import Tensor

from src.degradations.base import BaseDegradation


def shift_key(modality: str, step: int) -> str:
    return f"{modality}_shift_{step}"


class TemporalMisalignment(BaseDegradation):
    """Replaces `modality` with a non-corresponding patch `severity` steps away.

    reBEN has one S1/S2 pair per patch, so a step is a SPATIAL offset (pre-analysis §7.6
    fallback): step k pairs each optical patch with the SAR patch k grid cells away in the
    same tile (k * 1.2 km at 120 px and 10 m). src.prepare_data precomputes the shifted SAR
    into the test shards as `<modality>_shift_<k>`, because neighbours must come from the
    full source, not the subsampled tier. Severity must be one of the prepared steps.
    """

    def __init__(self, modality: str = "sar"):
        self.modality = modality

    def __call__(
        self, batch: dict[str, Tensor | None], severity: float, generator: torch.Generator
    ) -> dict[str, Tensor | None]:
        out = dict(batch)
        if severity == 0.0 or batch.get(self.modality) is None:
            return out
        step = int(severity)
        if step != severity:
            raise ValueError(f"temporal severity must be an integer step, got {severity}")
        key = shift_key(self.modality, step)
        if key not in batch:
            have = sorted(k for k in batch if k.startswith(f"{self.modality}_shift_"))
            raise KeyError(f"no {key} in the batch (prepared: {have}); re-run prepare_data "
                           f"with --temporal-steps including {step}")
        out[self.modality] = batch[key]
        return out
