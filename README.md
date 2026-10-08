# robustness-eval
Multimodal degradation-robustness experiments: a model, dataset or degradation is one file + one YAML.

## Install
`uv sync --extra dev` (Python 3.12, pinned by `uv.lock`; add `--extra logging` for W&B).

## Smoke test (CPU, no network)
```bash
uv run python -m src.train +experiment=smoke      # -> outputs/<run_id>/{checkpoints,run.jsonl}
uv run python -m src.evaluate run_dir=outputs/<run_id> degradation=gaussian_noise
uv run python -m src.aggregate outputs            # merged CSV + degradation curves
```
Every evaluation writes its own `outputs/<run_id>/results/<eval_id>.csv`; `aggregate` merges them.

## Add a model: `src/models/tiny_cnn.py`
```python
from torch import Tensor, nn
from src.models.base import BaseModel

class TinyCNN(BaseModel):
    def __init__(self, num_classes: int, in_channels: dict[str, int], image_size: int, width: int = 16):
        super().__init__()
        self.required_modalities, self.num_classes = ("optical",), num_classes
        self.net = nn.Sequential(nn.Conv2d(in_channels["optical"], width, 3, padding=1), nn.ReLU(),
                                 nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(width, num_classes))

    def forward(self, batch: dict[str, Tensor | None]) -> Tensor:  # raw logits, never sigmoid
        if (x := batch.get("optical")) is None:  # a dropped modality must not crash
            ref = next(v for k, v in batch.items() if k != "label" and v is not None)
            return ref.new_zeros(ref.shape[0], self.num_classes)
        return self.net(x)
```
and `configs/model/tiny_cnn.yaml`:
```yaml
_target_: src.models.tiny_cnn.TinyCNN
num_classes: ${data.num_classes}
in_channels: ${data.in_channels}
image_size: ${data.image_size}
```
Run: `uv run python -m src.train model=tiny_cnn +experiment=smoke`, or remotely `./scripts/submit_hf.sh model=tiny_cnn` / `./scripts/submit_kaggle.sh model=tiny_cnn`.
