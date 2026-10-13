import torch
from torch import Tensor, nn


class _Block(nn.Module):
    """ResNet basic block; `stride` 2 halves the resolution."""

    def __init__(self, cin: int, cout: int, stride: int):
        super().__init__()
        self.conv1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(cout)
        self.conv2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(cout)
        self.skip = (nn.Identity() if stride == 1 and cin == cout else
                     nn.Sequential(nn.Conv2d(cin, cout, 1, stride, bias=False),
                                   nn.BatchNorm2d(cout)))

    def forward(self, x: Tensor) -> Tensor:
        y = torch.relu(self.bn1(self.conv1(x)))
        return torch.relu(self.bn2(self.conv2(y)) + self.skip(x))


class ConvEncoder(nn.Module):
    """Small ResNet-style encoder: image -> feature map (B, out_dim, h, w).

    Shared initialisation across experiment arms (pre-analysis §6): the encoder is always
    built over the FULL band layout `bands` (e.g. {"optical": 12, "sar": 2}), initialised or
    loaded from `pretrained`, and only then is the stem sliced to `modalities`. So the
    optical encoder of `opt`, `late` and `gate_*` and the joint encoder of `early` all start
    from the same weights for a given seed; only the fusion layers and heads differ.

    Random init draws from a private generator seeded with `init_seed` (default: the global
    torch seed, i.e. the training seed), so it does not depend on what else the model
    builds or in which order.
    """

    def __init__(
        self,
        bands: dict[str, int],
        modalities: tuple[str, ...] | list[str],
        width: int = 32,
        depth: int = 4,
        pretrained: str | None = None,
        init_seed: int | None = None,
    ):
        super().__init__()
        self.modalities = tuple(modalities)
        unknown = set(self.modalities) - set(bands)
        if unknown:
            raise ValueError(f"modalities {sorted(unknown)} not in bands {list(bands)}")
        total = sum(bands.values())
        self.stem = nn.Sequential(nn.Conv2d(total, width, 3, 1, 1, bias=False),
                                  nn.BatchNorm2d(width), nn.ReLU())
        dims = [width * 2**i for i in range(depth)]
        self.blocks = nn.Sequential(*[_Block(cin, cout, 2)
                                      for cin, cout in zip([width] + dims[:-1], dims)])
        self.out_dim = dims[-1]

        if pretrained:
            state = torch.load(pretrained, map_location="cpu", weights_only=True)
            self.load_state_dict(state.get("state_dict", state))
        else:
            self._init(torch.initial_seed() if init_seed is None else init_seed)

        # Slice the full-band stem to this encoder's bands.
        offsets, start = {}, 0
        for m, c in bands.items():
            offsets[m] = (start, start + c)
            start += c
        idx = torch.cat([torch.arange(*offsets[m]) for m in self.modalities])
        conv = self.stem[0]
        sliced = nn.Conv2d(len(idx), width, 3, 1, 1, bias=False)
        with torch.no_grad():
            sliced.weight.copy_(conv.weight[:, idx])
        self.stem[0] = sliced
        self.in_channels = len(idx)

    def _init(self, seed: int) -> None:
        g = torch.Generator().manual_seed(seed % 2**63)
        for mod in self.modules():
            if isinstance(mod, nn.Conv2d):
                nn.init.kaiming_normal_(mod.weight, mode="fan_out", nonlinearity="relu",
                                        generator=g)
            elif isinstance(mod, nn.BatchNorm2d):
                nn.init.ones_(mod.weight)
                nn.init.zeros_(mod.bias)

    def forward(self, x: Tensor) -> Tensor:
        return self.blocks(self.stem(x))
