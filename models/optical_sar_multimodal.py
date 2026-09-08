
from typing import List, Sequence

import torch
import torch.nn as nn

from models.decoder import UNetDecoder
from models.encoders import CNNEncoder
from models.fusion import TemporalInteraction, build_fusion


class _TemporalBranch(nn.Module):
    """Modality-specific encoder + per-scale temporal interaction."""

    def __init__(self, in_channels, widths, blocks, out_strides, temporal_mode):
        super().__init__()
        self.encoder = CNNEncoder(in_channels, tuple(widths), tuple(blocks), tuple(out_strides))
        self.temporal = nn.ModuleList([TemporalInteraction(c, temporal_mode) for c in widths])

    def forward(self, t1: torch.Tensor, t2: torch.Tensor) -> List[torch.Tensor]:
        f1, f2 = self.encoder(t1), self.encoder(t2)
        return [m(a, b) for m, a, b in zip(self.temporal, f1, f2)]


class OpticalSARChangeNet(nn.Module):
    """Inputs:
        opt_t1, opt_t2: [B, C_opt, H, W]
        sar_t1, sar_t2: [B, C_sar, H, W]   (C_sar may differ from C_opt)
    All four tensors must share H, W (divisible by max(out_strides), default 32).
    Output: logits [B, 1, H, W].
    """

    def __init__(
        self,
        optical_channels=3,
        sar_channels=2,
        encoder_widths=(64, 128, 256),
        encoder_blocks=(2, 2, 2),
        out_strides=(8, 16, 32),
        decoder_channels=256,
        temporal_mode="difference",
        fusion_mode="cross_attention",
        fusion_heads=4,
    ):
        super().__init__()
        self.optical_branch = _TemporalBranch(
            optical_channels,
            encoder_widths,
            encoder_blocks,
            out_strides,
            temporal_mode,
        )
        self.sar_branch = _TemporalBranch(
            sar_channels,
            encoder_widths,
            encoder_blocks,
            out_strides,
            temporal_mode,
        )
        self.fusion = nn.ModuleList(
            [build_fusion(fusion_mode, c, num_heads=fusion_heads) for c in encoder_widths]
        )
        self.decoder = UNetDecoder(tuple(encoder_widths), tuple(out_strides), decoder_channels)

    def forward(
        self,
        opt_t1: torch.Tensor,
        opt_t2: torch.Tensor,
        sar_t1: torch.Tensor,
        sar_t2: torch.Tensor,
    ) -> torch.Tensor:
        assert opt_t1.shape == opt_t2.shape and sar_t1.shape == sar_t2.shape, "T1/T2 shape mismatch"
        assert opt_t1.shape[-2:] == sar_t1.shape[-2:], (
            f"optical/SAR spatial mismatch: {opt_t1.shape[-2:]} vs {sar_t1.shape[-2:]}"
        )
        d_opt = self.optical_branch(opt_t1, opt_t2)
        d_sar = self.sar_branch(sar_t1, sar_t2)
        fused = [f(o, s) for f, o, s in zip(self.fusion, d_opt, d_sar)]
        return self.decoder(fused)
