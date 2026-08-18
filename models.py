"""Model components: CNN sensor encoder, projector, classification heads."""

import torch
from torch import nn

ENCODER_DIM = 256
LLM_DIM = 960  # SmolLM2-360M hidden size
NUM_CLASSES = 6


class SensorEncoder(nn.Module):
    """1D CNN over a (batch, 9, 128) window -> (batch, 256) vector.

    Three conv blocks with stride-2 downsampling (128 -> 64 -> 32 -> 16),
    then global average pooling.
    """

    def __init__(self) -> None:
        super().__init__()
        channels = [9, 64, 128, ENCODER_DIM]
        blocks = []
        for c_in, c_out in zip(channels[:-1], channels[1:]):
            blocks += [
                nn.Conv1d(c_in, c_out, kernel_size=7, stride=2, padding=3, bias=False),
                nn.BatchNorm1d(c_out),
                nn.ReLU(inplace=True),
            ]
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(x).mean(dim=2)


class Projector(nn.Module):
    """256 -> 960 MLP whose output is scaled to the LLM embedding regime.

    Ends in LayerNorm, a fixed multiply by the frozen embedding table's mean
    L2 norm, and a learnable scalar gain initialized to 1. LayerNorm output
    has L2 norm ~sqrt(dim), hence the sqrt(dim) division.
    """

    def __init__(self, target_norm: float) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(ENCODER_DIM, LLM_DIM),
            nn.GELU(),
            nn.Linear(LLM_DIM, LLM_DIM),
            nn.LayerNorm(LLM_DIM),
        )
        self.register_buffer("scale", torch.tensor(target_norm / LLM_DIM**0.5))
        self.gain = nn.Parameter(torch.tensor(1.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x) * self.scale * self.gain


class DirectClassifier(nn.Module):
    """Required baseline: encoder -> linear head."""

    def __init__(self) -> None:
        super().__init__()
        self.encoder = SensorEncoder()
        self.head = nn.Linear(ENCODER_DIM, NUM_CLASSES)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(x))


class CapacityMatchedClassifier(nn.Module):
    """Ablation: encoder -> projector-shaped MLP -> head.

    Matches the context model's trainable parameter count so any context-model
    gain can be attributed to the LLM rather than extra capacity.
    """

    def __init__(self) -> None:
        super().__init__()
        self.encoder = SensorEncoder()
        self.projector = Projector(target_norm=1.0)
        self.head = nn.Linear(LLM_DIM, NUM_CLASSES)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.projector(self.encoder(x)))


def count_trainable(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


if __name__ == "__main__":
    direct = DirectClassifier()
    matched = CapacityMatchedClassifier()

    x = torch.randn(4, 9, 128)
    assert direct(x).shape == (4, NUM_CLASSES)
    assert matched(x).shape == (4, NUM_CLASSES)

    encoder = count_trainable(direct.encoder)
    projector = count_trainable(matched.projector)
    context_head = NUM_CLASSES * (LLM_DIM + 1)
    print(f"encoder:              {encoder:>9,}")
    print(f"projector:            {projector:>9,}")
    print(f"direct head:          {count_trainable(direct.head):>9,}")
    print(f"context head:         {context_head:>9,}")
    print(f"direct total:         {count_trainable(direct):>9,}")
    print(f"context total:        {encoder + projector + context_head:>9,}")
    print(f"capacity-matched:     {count_trainable(matched):>9,}")
    print(f"budget:              10,000,000")
