
from typing import Sequence

import torch
import torch.nn as nn

from models.decoder import UNetDecoder
from models.encoders import CNNEncoder
from models.fusion import TemporalInteraction


class SiameseTemporalCD(nn.Module):
    """Inputs: t1, t2 of shape [B, C, H, W]. H, W must be divisible by
    max(out_strides) (default 32; use out_strides=(4, 8, 16) for stride-16).
    Output: logits [B, 1, H, W] (apply sigmoid for probabilities).

    Weights are shared between the two timestamps (one encoder called twice),
    so both epochs live in the same feature space before differencing.
    """

    def __init__(
        self,
        in_channels: int = 3,
        encoder_widths: Sequence[int] = (64, 128, 256, 512),
        encoder_blocks: Sequence[int] = (2, 2, 2, 2),
        out_strides: Sequence[int] = (4, 8, 16, 32),
        decoder_channels: int = 256,
        temporal_mode: str = "difference",  # 'difference' | 'attention'
    ):
        super().__init__()
        self.encoder = CNNEncoder(in_channels, tuple(encoder_widths), tuple(encoder_blocks), tuple(out_strides))
        self.temporal = nn.ModuleList([TemporalInteraction(c, temporal_mode) for c in encoder_widths])
        self.decoder = UNetDecoder(tuple(encoder_widths), tuple(out_strides), decoder_channels)

    def forward(self, t1: torch.Tensor, t2: torch.Tensor) -> torch.Tensor:
        assert t1.shape == t2.shape, f"T1/T2 shape mismatch: {t1.shape} vs {t2.shape}"
        f1 = self.encoder(t1)
        f2 = self.encoder(t2)
        diffs = [m(a, b) for m, a, b in zip(self.temporal, f1, f2)]
        return self.decoder(diffs)
